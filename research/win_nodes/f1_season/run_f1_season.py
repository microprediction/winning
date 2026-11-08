"""A season as one Gaussian network: pace per driver per weekend under a
random-walk prior, a team factor within each weekend, and the weekend's
evidence folded in through win nodes. Scored before each race by the log
probability given to the actual winner, and before each qualifying by
the log probability given to the pole sitter.

Variants: 'race' (classification only), 'race+quali' (Q1 set, Q2 set with
the Q1 casualties withdrawn, Q3 order as a prefix ahead of the rest),
'race+quali+gaps' (Q3 gaps as margins in place of the order).

Usage: OMP_NUM_THREADS=1 python run_f1_season.py 2024 [variant ...]
"""
import json, os, sys, time
import numpy as np

from winning.ratings.net import GaussianNet
from winning.ratings.tracker import block_loadings
from f1db_load import load_season

YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
VARIANTS = sys.argv[2:] or ["race", "race+quali", "race+market", "race+quali+market"]
_e = os.environ.get
INIT_VAR, DRIFT, RHO = float(_e("INIT_VAR", 1.0)), float(_e("DRIFT", 0.05)), float(_e("RHO", 0.5))
BETA2_Q, BETA2_R, LS_Q = float(_e("BETA2_Q", 0.5)), float(_e("BETA2_R", 1.0)), float(_e("LS_Q", 2.0))
NODES, TAU2 = int(_e("NODES", 6)), float(_e("TAU2", 0.3))
TAG = _e("TAG", "")
SPLIT_QR = float(_e("SPLIT_QR", 0.0))
QUALI_MODE = _e("QUALI_MODE", "prefix")    # prefix | sets (Q1 and Q2 sets only, no ordering)
Q_BY_TIMES = bool(_e("Q_BY_TIMES", ""))     # f1db_load: time-based classification (the old reading)
BELIEF = _e("BELIEF", "dense")              # dense: GaussianNet over the season | diag: per-weekend update_race with drift
WATCH = [d for d in _e("WATCH", "").split(",") if d]      # > 0: separate qualifying and race pace, correlated by this much
from f1_market_join import snapshots as market_snapshots
HERE = os.path.dirname(os.path.abspath(__file__))


class DiagNet:
    """The tracker-style belief: one (mean, var) per driver, random-walk drift
    between weekends, every message through update_race on the weekend's
    members. Same attach/observe/query surface as GaussianNet for run()."""
    def __init__(self):
        from winning.ratings.market import update_race
        self._up = update_race
        self.mean, self.var, self.last = {}, {}, {}
        self.nodes, self.evidence = {}, 0.0
        self.index = {}

    def _gather(self, drivers, rnd):
        m = np.zeros(len(drivers)); v = np.zeros(len(drivers))
        for i, d in enumerate(drivers):
            if d in self.mean:
                dt = rnd - self.last[d]
                m[i], v[i] = self.mean[d], self.var[d] + DRIFT * dt
            else:
                m[i], v[i] = 0.0, INIT_VAR
        return m, v

    def attach(self, name, members, beta2, V):
        drivers = [x.split("@")[0] for x in members]; rnd = int(members[0].split("@")[1].split(":")[0])
        self.nodes[name] = dict(drivers=drivers, rnd=rnd, beta2=beta2, V=V, labels=list(members))
        return name

    def query(self, name, k=1):
        from winning.factor.races import race_probabilities
        nd = self.nodes[name]; m, v = self._gather(nd["drivers"], nd["rnd"])
        return np.asarray(race_probabilities(-m, V=nd["V"], D=v + nd["beta2"]))

    def observe(self, name, nodes_log2=10, dnf=None, prices=None, tau2=0.25, **kw):
        nd = self.nodes[name]; drivers, rnd = nd["drivers"], nd["rnd"]
        pos = {lab: i for i, lab in enumerate(nd["labels"])}
        m, v = self._gather(drivers, rnd)
        conv = lambda labs: [pos[x] for x in labs] if labs is not None else None
        kw2 = {}
        for key, val in kw.items():
            if key in ("prefix", "top", "order", "tie"):
                kw2[key] = conv(val)
            elif key == "member_of_top":
                kw2[key] = (pos[val[0]], val[1])
            elif key in ("margins", "scores"):
                kw2[key] = np.asarray(val, float)
            else:
                kw2[key] = val
        if prices is not None:
            kw2["p_market"] = np.asarray(prices, float); kw2["tau2"] = tau2
        m, v, info = self._up(m, v, V=nd["V"], beta2=nd["beta2"], nodes_log2=nodes_log2, dnf=conv(dnf), **kw2)
        for i, d in enumerate(drivers):
            self.mean[d], self.var[d], self.last[d] = float(m[i]), float(v[i]), rnd
        self.evidence += sum(x for x in info.values())
        return info


def build(season):
    if BELIEF == "diag":
        return DiagNet()
    names, blocks = [], []
    appearances = {}
    for w in season:
        for d in w["drivers"]:
            appearances.setdefault(d, []).append(w["round"])
    for d, rounds in appearances.items():
        r = np.asarray(rounds, float)
        RW = INIT_VAR + DRIFT * np.minimum.outer(r, r)
        if SPLIT_QR > 0:
            # two paces per weekend, qualifying and race, correlated SPLIT_QR,
            # each a random walk across the season
            names += [f"{d}@{k}:{s}" for k in rounds for s in ("q", "r")]
            blocks.append(np.kron(RW, np.array([[1.0, SPLIT_QR], [SPLIT_QR, 1.0]])))
        else:
            names += [f"{d}@{k}" for k in rounds]
            blocks.append(RW)
    n = len(names)
    S = np.zeros((n, n)); i = 0
    for B in blocks:
        k = len(B); S[i:i + k, i:i + k] = B; i += k
    return GaussianNet(names, np.zeros(n), S)


def run(variant, season, snaps):
    net = build(season)
    pole_ls, win_ls, t0 = [], [], time.time()
    mkt_ls, win_ls_on_mkt = [], []        # the market's own log score, and ours, on the races with a pre-race snapshot
    rounds_log = open(os.path.join(HERE, f"rounds_{YEAR}{TAG}_{variant.replace('+', '_')}.jsonl"), "a")
    for w in season:
        r, drv = w["round"], w["drivers"]
        pre = [s for s in snaps.get(r, []) if s[0] == "pre_quali"]
        post = [s for s in snaps.get(r, []) if s[0] == "post_quali"]

        def price_in(snapshot):
            _, _, probs = snapshot
            quoted = [d for d in drv if d in probs]
            if len(quoted) < 3 or w["winner"] not in probs:
                return
            p = np.array([probs[d] for d in quoted]); p = p / p.sum()
            teams_q = sorted(set(w["team"][d] for d in quoted)); g = [teams_q.index(w["team"][d]) for d in quoted]
            Vm, b2m = block_loadings(g, RHO, BETA2_R)
            node = net.attach(f"mkt{r}_{snapshot[1]}", members=[rlab[d] for d in quoted], beta2=b2m, V=Vm)
            net.observe(node, prices=p, tau2=TAU2, nodes_log2=NODES)
        qmem = [f"{d}@{r}:q" if SPLIT_QR > 0 else f"{d}@{r}" for d in drv]
        members = [f"{d}@{r}:r" if SPLIT_QR > 0 else f"{d}@{r}" for d in drv]
        teams = sorted(set(w["team"].values())); groups = [teams.index(w["team"][d]) for d in drv]
        Vq, b2q = block_loadings(groups, RHO, BETA2_Q); Vr, b2r = block_loadings(groups, RHO, BETA2_R)
        qn = net.attach(f"q{r}", members=qmem, beta2=b2q, V=Vq)
        rn = net.attach(f"race{r}", members=members, beta2=b2r, V=Vr)
        lab = {d: f"{d}@{r}:q" if SPLIT_QR > 0 else f"{d}@{r}" for d in drv}        # qualifying labels
        rlab = {d: f"{d}@{r}:r" if SPLIT_QR > 0 else f"{d}@{r}" for d in drv}       # race labels
        if "market" in variant:
            for s_ in sorted(pre, key=lambda s: s[1]):
                price_in(s_)
        if "quali" in variant and w["q3_order"]:
            p = net.query(qn); pole_ls.append(float(np.log(p[drv.index(w["q3_order"][0])])))
            nq = [lab[d] for d in w["no_quali"]]                 # took no part in qualifying
            net.observe(qn, top=[lab[d] for d in drv if d not in w["q1_out"] and d not in w["no_quali"]],
                        dnf=nq or None, nodes_log2=NODES)
            q1 = nq + [lab[d] for d in w["q1_out"]]
            if "gaps" in variant:
                net.observe(qn, top=[lab[d] for d in w["q3_order"]], dnf=q1, nodes_log2=NODES)
                q12 = q1 + [lab[d] for d in w["q2_out"]]
                margins = np.array([w["q3_gap_s"].get(d, 0.0) for d in drv])
                net.observe(qn, margins=margins, dnf=q12, lengths_scale=LS_Q, nodes_log2=NODES)
            elif QUALI_MODE == "sets":
                net.observe(qn, top=[lab[d] for d in drv if d not in w["q1_out"] and d not in w["q2_out"] and d not in w["no_quali"]],
                            dnf=q1, nodes_log2=NODES)
            else:
                # Q3 participants without a time are in the top ten but unordered:
                # withdrawn from the prefix message rather than placed behind
                untimed = [lab[d] for d in w.get("q3_untimed", [])]
                net.observe(qn, prefix=[lab[d] for d in w["q3_order"]], dnf=q1 + untimed, nodes_log2=NODES)
        if "market" in variant:
            for s_ in sorted(post, key=lambda s: s[1]):
                price_in(s_)
        if w["winner"]:
            p = net.query(rn); win_ls.append(float(np.log(p[drv.index(w["winner"])])))
            last = (post or pre)
            if last:
                probs = sorted(last, key=lambda s: s[1])[-1][2]
                if w["winner"] in probs:
                    quoted = [d for d in drv if d in probs]; pm = np.array([probs[d] for d in quoted]); pm /= pm.sum()
                    mkt_ls.append(float(np.log(pm[quoted.index(w["winner"])])))
                    win_ls_on_mkt.append(float(np.log(p[drv.index(w["winner"])])))
            net.observe(rn, order=[rlab[d] for d in w["finishers"]], dnf=[rlab[d] for d in w["dnf"]], nodes_log2=NODES)
        watch = {d: (float(net.m[net.index[rlab[d]]]), float(np.sqrt(net.S[net.index[rlab[d]], net.index[rlab[d]]])))
                 for d in WATCH if d in drv}
        rounds_log.write(json.dumps({"round": r, "name": w["name"], "watch_mean_sd_after": watch,
                                     "params": PARAMS, "variant": variant, "pole_log_score": pole_ls[-1] if pole_ls and "quali" in variant else None,
                                     "winner": w["winner"], "winner_log_score": win_ls[-1] if win_ls else None,
                                     "market_log_score": mkt_ls[-1] if mkt_ls and (post or pre) and w["winner"] in sorted(post or pre, key=lambda s: s[1])[-1][2] else None,
                                     "evidence_so_far": float(net.evidence)}) + "\n"); rounds_log.flush()
        print(f"  {variant:16s} round {r:2d} done {time.time() - t0:6.1f}s", flush=True)
    rounds_log.close()
    return {"variant": variant, "n_races": len(win_ls), "mean_log_score_winner": float(np.mean(win_ls)),
            "total_evidence": float(net.evidence), "belief": BELIEF,
            "races_with_market": len(mkt_ls), "market_log_score": float(np.mean(mkt_ls)) if mkt_ls else None,
            "model_log_score_on_those": float(np.mean(win_ls_on_mkt)) if win_ls_on_mkt else None,
            "mean_log_score_pole": float(np.mean(pole_ls)) if pole_ls else None, "split_qr": SPLIT_QR,
            "uniform": float(np.log(1 / 20)), "seconds": time.time() - t0}


PARAMS = dict(init_var=INIT_VAR, drift=DRIFT, rho=RHO, beta2_q=BETA2_Q, beta2_r=BETA2_R, lengths_scale_q=LS_Q,
              nodes_log2=NODES, tau2=TAU2, belief=BELIEF, quali_mode=QUALI_MODE, q_by_times=Q_BY_TIMES, split_qr=SPLIT_QR)


def main():
    t_start = time.time()
    season = load_season(YEAR)
    snaps = market_snapshots(YEAR, {d for w in season for d in w["drivers"]})
    out = [run(v, season, snaps) for v in VARIANTS]
    # one summary file per run (year, tag, variants, start time): a later run
    # never overwrites an earlier one
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(t_start))
    path = os.path.join(HERE, f"season_{YEAR}{TAG}_{'_'.join(v.replace('+', '-') for v in VARIANTS)}_{stamp}.json")
    with open(path, "w") as f:
        json.dump({"year": YEAR, "params": PARAMS, "variants": VARIANTS, "results": out}, f, indent=1)
    print("summary ->", path)
    for o in out:
        print(f"{o['variant']:20s} evidence {o['total_evidence']:.1f} winner log score {o['mean_log_score_winner']:.3f}"
              f"  on {o['races_with_market']} market races: model {o['model_log_score_on_those']} vs market {o['market_log_score']}"
              f"  pole {o['mean_log_score_pole'] if o['mean_log_score_pole'] is None else round(o['mean_log_score_pole'], 3)}"
              f"  (uniform {o['uniform']:.3f})  {o['seconds']:.0f}s")


if __name__ == "__main__":
    main()
