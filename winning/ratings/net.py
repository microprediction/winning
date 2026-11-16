"""A Gaussian network with win nodes: one joint Gaussian belief over any
set of named variables, and order-statistic nodes attached to any linear
map of them.

The variables can be anything Gaussian -- abilities, abilities at each
date, each team's goals in each half, each firm's assets at each
monitoring date. A node is a k x n matrix A taking the belief into an
observation space (rows select members, or sum a player's rounds), with
its own performance noise beta2 and optional shared factors V. Every
kind of evidence the race front door takes is accepted on any node, and
each one updates the FULL belief by the exact-moment Gaussian update, so
evidence on one node moves every correlated variable; a node's
probabilities are read off the current belief at any time.

Forward: the node's predictive covariance A S A' + V V' + diag(beta2) is
fitted to the factor grammar and priced by the shared-field pass.
Backward: the dense updates of ratings.full / teams (winner, order,
margins, prices) and the place messages of ratings.nway (first k as a
set, first k in order, membership, dead heat) through the same mixture
with A. Gaussian projection after a non-Gaussian message is the one
approximation, as in expectation propagation; prices are conjugate.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from ..shapes import as_loadings
from .full import _mixture_update_full, _psd_repair
from .nway import _topset_logp_grad, _prefix_logp, _tie_logdens_grad, _beta_per_player
from .teams import (update_team_winner_full, update_team_order_full,
                    update_team_margins_full, update_team_market_full)


def random_walk_cov(T, init_var=1.0, drift=0.05):
    """Prior covariance of one ability followed over T dates under a
    random walk: Cov(t, u) = init_var + drift * min(t, u)."""
    t = np.arange(T)
    return init_var + drift * np.minimum.outer(t, t).astype(float)


class WinNode:
    """A node: rows of A pick (or combine) the variables, beta2 the
    performance noise per row, V optional shared factors per row."""

    def __init__(self, name, A, beta2=1.0, V=None, labels=None):
        self.name = name
        self.A = np.atleast_2d(np.asarray(A, dtype=float))
        k = self.A.shape[0]
        self.beta2 = _beta_per_player(beta2, k)
        self.V = None if V is None else as_loadings(V, k)
        self.labels = list(labels) if labels is not None else list(range(k))

    @property
    def k(self):
        return self.A.shape[0]

    def restricted(self, keep):
        """The node on a subset of its rows (a field with withdrawals)."""
        keep = list(keep)
        return WinNode(self.name, self.A[keep], self.beta2[keep],
                       None if self.V is None else self.V[keep], [self.labels[i] for i in keep])


class GaussianNet:
    def __init__(self, names: Sequence, mean, cov):
        self.names = list(names)
        self.index: Dict = {nm: i for i, nm in enumerate(self.names)}
        if len(self.index) != len(self.names):
            raise ValueError("variable names must be distinct")
        n = len(self.names)
        self.m = np.asarray(mean, dtype=float).copy()
        S = np.asarray(cov, dtype=float)
        if self.m.shape != (n,) or S.shape != (n, n):
            raise ValueError(f"mean must be ({n},) and cov ({n}, {n}); got {self.m.shape} and {S.shape}")
        self.S = _psd_repair(S.copy())
        self.nodes: Dict[str, WinNode] = {}
        self.evidence = 0.0

    # --- building
    def attach(self, name, members=None, A=None, beta2=1.0, V=None):
        """Attach a node over `members` (variable names, one row each) or
        over an explicit k x n matrix A (rows may sum variables)."""
        if (members is None) == (A is None):
            raise ValueError("give members= or A=, not both")
        if A is None:
            rows = [self.index[v] for v in members]
            A = np.zeros((len(rows), len(self.names)))
            A[np.arange(len(rows)), rows] = 1.0
            labels = list(members)
        else:
            labels = None
        node = WinNode(name, A, beta2=beta2, V=V, labels=labels)
        if node.A.shape[1] != len(self.names):
            raise ValueError(f"A must have {len(self.names)} columns")
        self.nodes[name] = node
        return node

    # --- reading
    def predictive(self, node):
        nd = self.nodes[node] if isinstance(node, str) else node
        mu = nd.A @ self.m
        C = nd.A @ self.S @ nd.A.T + np.diag(nd.beta2)
        if nd.V is not None:
            C = C + nd.V @ nd.V.T
        return mu, _psd_repair(C)

    def split(self, node):
        """The node's predictive as loadings plus idiosyncratic variance,
        (mu, B, D): the belief part of the covariance is split numerically
        by _belief_split (diagonal -> no factors; rank-r-plus-diagonal ->
        r factors; else the exact square root), and the node's own
        factors V are appended. This is where structure is DETECTED: a
        diagonal belief prices on the lattice alone, exactly."""
        from .full import _belief_split
        from .nway import _canonical_loadings
        nd = self.nodes[node] if isinstance(node, str) else node
        mu = nd.A @ self.m
        B, psi = _belief_split(_psd_repair(nd.A @ self.S @ nd.A.T))
        D = psi + nd.beta2
        if nd.V is not None:
            B = np.hstack([B, nd.V])
        if B.shape[1] == 0:
            return mu, None, D
        # the quadrature rule is not rotation invariant, so the loadings are
        # canonicalised as the backward update canonicalises them (#130):
        # an eigendecomposition that differs between machines must not
        # move the forward answer. Measured: 1e-4 across platforms before.
        return mu, _canonical_loadings(B), D

    def query(self, node, k=1):
        """P(each member wins), or P(each member in the first k), from the
        current belief."""
        from ..factor.races import race_probabilities
        from ..factor.topk import top_k_probabilities
        mu, B, D = self.split(node)
        if k == 1:
            return np.asarray(race_probabilities(-mu, V=B, D=D))      # the engine speaks min-wins
        return np.asarray(top_k_probabilities(-mu, k, V=B, D=D))

    # --- evidence
    def observe(self, node, winner=None, order=None, prefix=None, top=None, member_of_top=None,
                tie=None, margins=None, scores=None, prices=None, tau2=0.25, dnf=None,
                lengths_scale=1.0, meas_var=0.0, nodes_log2=10, **market_model):
        """Fold one piece of evidence on `node`. Labels are the node's
        member labels (variable names when attached by members=, row
        indices otherwise). Prices may accompany one outcome and are
        applied first; `dnf` removes rows from the outcome's field."""
        nd = self.nodes[node] if isinstance(node, str) else node
        given = [k for k, x in (("winner", winner), ("order", order), ("prefix", prefix), ("top", top),
                                ("member_of_top", member_of_top), ("tie", tie), ("margins", margins),
                                ("scores", scores)) if x is not None]
        if len(given) > 1:
            raise ValueError(f"give one outcome, not {given}: they are not independent evidence")
        from ..outcomes import as_order
        pos = {lab: i for i, lab in enumerate(nd.labels)}
        info = {}
        # --- validate EVERYTHING before any state changes, so a bad outcome
        # beside a price cannot leave the belief half-updated (review of #1026)
        if prices is not None:
            pr = np.asarray(prices, dtype=float)
            if pr.shape != (nd.k,) or not np.all(np.isfinite(pr)):
                raise ValueError(f"prices must be one finite entry per member of the node; got shape {pr.shape} for {nd.k}")
        field = nd
        if dnf is not None and given:
            bad = [x for x in dnf if x not in pos]
            if bad:
                raise ValueError(f"dnf names {bad[0]!r}, which is not in the field")
            out = {pos[x] for x in dnf}
            keep = [i for i in range(nd.k) if i not in out]
            if len(keep) < 2:
                raise ValueError("dnf leaves fewer than two entrants in the outcome")
            field = nd.restricted(keep)
            pos = {lab: i for i, lab in enumerate(field.labels)}

        def rel(labels, what, lo=1, hi=None):
            """labels -> row indices of the field, distinct and in range, with
            between lo and hi of them (the contract the diagonal verbs apply)"""
            try:
                idx = [pos[x] for x in labels]
            except KeyError as e:
                raise ValueError(f"{what} names {e.args[0]!r}, which is not in the field") from None
            idx = [int(i) for i in as_order(idx, field.k, name=what, min_len=lo)]
            if hi is not None and len(idx) > hi:
                raise ValueError(f"{what} names {len(idx)} entrants; at most {hi} of the {field.k} in the field")
            return idx
        if winner is not None:
            w_idx = rel([winner], "winner")[0]
        if order is not None:
            o_idx = rel(order, "order")
        if prefix is not None:
            p_idx = rel(prefix, "prefix")                      # 1..k, in order, rest behind
        if top is not None:
            t_idx = rel(top, "top", hi=field.k - 1)            # a set needs at least one outsider
        if tie is not None:
            tie_idx = rel(tie, "tie")                          # 1 is the winner
        if member_of_top is not None:
            mo_idx = rel([member_of_top[0]], "member_of_top")[0]
            if not 1 <= int(member_of_top[1]) <= field.k:
                raise ValueError(f"member_of_top k must lie in 1..{field.k}; got {member_of_top[1]}")
        if margins is not None or scores is not None:
            arr = np.asarray(margins if margins is not None else scores, dtype=float)
            if arr.shape != (nd.k,) or not np.all(np.isfinite(arr[[i for i, lab in enumerate(nd.labels) if lab in pos]])):
                raise ValueError(f"margins/scores must have one finite entry per member of the node; got {arr.shape} for {nd.k}")
        # --- now the updates
        if prices is not None:
            self.m, self.S, lz = update_team_market_full(self.m, self.S, nd.A, prices, tau2=tau2,
                                                         **({"V": nd.V, "D": nd.beta2} | market_model))
            self.evidence += lz; info["logZ_market"] = lz
        if not given:
            return info
        A, V, b2 = field.A, field.V, field.beta2
        if winner is not None:
            self.m, self.S, lz = update_team_winner_full(self.m, self.S, A, w_idx,
                                                         V=V, beta2=b2, nodes_log2=nodes_log2)
        elif order is not None:
            self.m, self.S, lz = update_team_order_full(self.m, self.S, A, o_idx,
                                                        V=V, beta2=b2, nodes_log2=nodes_log2)
        elif margins is not None or scores is not None:
            if dnf is not None:
                arr = arr[[i for i, lab in enumerate(nd.labels) if lab in pos]]
            kw = {"margins": arr} if margins is not None else {"scores": arr}
            self.m, self.S, lz = update_team_margins_full(self.m, self.S, A, V=V, beta2=b2,
                                                          lengths_scale=lengths_scale, meas_var=meas_var, **kw)
        else:
            if prefix is not None:
                def node_fn(mm, D, _l=p_idx):
                    lp = _prefix_logp(mm, D, _l)
                    g = np.empty(len(mm)); h = 1e-5 * np.sqrt(D)
                    for j in range(len(mm)):
                        e = np.zeros(len(mm)); e[j] = h[j]
                        g[j] = (_prefix_logp(mm + e, D, _l) - _prefix_logp(mm - e, D, _l)) / (2 * h[j])
                    return lp, g
            elif top is not None:
                def node_fn(mm, D, _l=t_idx):
                    return _topset_logp_grad(mm, D, _l)
            elif tie is not None:
                def node_fn(mm, D, _l=tie_idx):
                    return _tie_logdens_grad(mm, D, _l)
            else:
                from ..factor.topk import top_k_probabilities, top_k_jacobian_row
                def node_fn(mm, D, _i=mo_idx, _k=int(member_of_top[1])):
                    mu = -np.asarray(mm, float)
                    q = max(float(np.asarray(top_k_probabilities(mu, _k, D=D))[_i]), 1e-300)
                    return np.log(q), -np.asarray(top_k_jacobian_row(mu, _i, _k, D=D)) / q
            self.m, self.S, lz = _mixture_update_full(self.m, self.S, V, b2, node_fn, A=A,
                                                      nodes_log2=nodes_log2)
        self.evidence += lz; info["logZ_outcome"] = lz
        return info
