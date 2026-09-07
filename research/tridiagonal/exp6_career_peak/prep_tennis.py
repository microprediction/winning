"""When was a player at their peak? The Bayes-net side: a latent
ability PATH with win evidence, reduced to a Gauss-Markov chain.

Model. Performances are unit-normal about ability, so player i beats
player j with probability Phi(a_i - a_j) on the difference scale.
Stage one fits static abilities for the whole field by MAP under a
standard normal prior (concave). Stage two takes one player, replaces
that player's constant by a yearly path theta_1..theta_T, holds the
field fixed, and puts a random-walk prior on the path:

    theta_1 ~ N(a_p, v0),   theta_{t+1} - theta_t ~ N(0, tau^2).

The log-likelihood is a sum of log Phi terms, each touching ONE year,
so its Hessian is diagonal; the random-walk prior's precision is
tridiagonal; the Laplace posterior precision is their sum and is
therefore TRIDIAGONAL. That is a Gauss-Markov chain, which is what
GMRFExtremes reads with chain_from_precision.

tau is chosen by the Laplace marginal likelihood over a grid.

Run:  python prep_tennis.py <out.json> [first_year] [last_year]
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr, ndtr

URL = ("https://raw.githubusercontent.com/VictorSquidWei/tennis_atp/"
       "master/atp_matches_{year}.csv")
CACHE = os.path.expanduser("~/.cache/winning/atp")
PLAYERS = ["Roger Federer", "Rafael Nadal", "Novak Djokovic",
           "Pete Sampras", "Andre Agassi", "Andy Murray"]


def load(first_year, last_year):
    os.makedirs(CACHE, exist_ok=True)
    winners, losers, years = [], [], []
    for y in range(first_year, last_year + 1):
        path = os.path.join(CACHE, f"atp_matches_{y}.csv")
        if not os.path.exists(path):
            try:
                urllib.request.urlretrieve(URL.format(year=y), path)
            except Exception:
                continue
        with open(path, encoding="utf-8", errors="ignore") as f:
            header = f.readline().rstrip("\n").split(",")
            try:
                wi, li = header.index("winner_name"), header.index("loser_name")
            except ValueError:
                continue
            for line in f:
                p = line.rstrip("\n").split(",")
                if len(p) <= max(wi, li):
                    continue
                w, l = p[wi].strip(), p[li].strip()
                if not w or not l or w == l:
                    continue
                winners.append(w)
                losers.append(l)
                years.append(y)
    return winners, losers, np.array(years)


def static_abilities(winners, losers, min_matches=15):
    """MAP Bradley-Terry on the probit scale: P(i beats j) = Phi(a_i - a_j),
    a ~ N(0, 1). Concave, so L-BFGS on the exact gradient suffices."""
    counts = {}
    for w, l in zip(winners, losers):
        counts[w] = counts.get(w, 0) + 1
        counts[l] = counts.get(l, 0) + 1
    names = sorted(n for n, c in counts.items() if c >= min_matches)
    idx = {n: i for i, n in enumerate(names)}
    keep = [k for k, (w, l) in enumerate(zip(winners, losers))
            if w in idx and l in idx]
    wi = np.array([idx[winners[k]] for k in keep])
    li = np.array([idx[losers[k]] for k in keep])
    n = len(names)

    def negll(a):
        z = a[wi] - a[li]
        ll = log_ndtr(z).sum() - 0.5 * (a @ a)
        lam = np.exp(-0.5 * z * z - 0.5 * np.log(2 * np.pi) - log_ndtr(z))
        g = np.zeros(n)
        np.add.at(g, wi, lam)
        np.add.at(g, li, -lam)
        return -ll, -(g - a)

    res = minimize(negll, np.zeros(n), jac=True, method="L-BFGS-B",
                   options={"maxiter": 400})
    return names, idx, res.x, np.array(keep), wi, li


def _lam(z):
    return np.exp(-0.5 * z * z - 0.5 * np.log(2 * np.pi) - log_ndtr(z))


def path_posterior(z_years, z_sign, opp, a0, tau, v0=1.0, iters=60):
    """MAP path and its (tridiagonal) Laplace precision.

    z_years: year index of each match; z_sign: +1 if the focal player
    won; opp: opponent ability. The match term is log Phi(s (theta_t -
    a_opp)), so gradient and Hessian both hit one coordinate."""
    T = z_years.max() + 1
    Qp = np.zeros((T, T))
    Qp[0, 0] = 1.0 / v0
    for t in range(T - 1):
        Qp[t, t] += 1.0 / tau ** 2
        Qp[t + 1, t + 1] += 1.0 / tau ** 2
        Qp[t, t + 1] -= 1.0 / tau ** 2
        Qp[t + 1, t] -= 1.0 / tau ** 2
    m = np.full(T, a0)

    theta = np.full(T, a0)
    for _ in range(iters):
        z = z_sign * (theta[z_years] - opp)
        lam = _lam(z)
        g = np.zeros(T)
        np.add.at(g, z_years, z_sign * lam)
        h = np.zeros(T)
        np.add.at(h, z_years, lam * (z + lam))       # positive curvature
        grad = g - Qp @ (theta - m)
        H = np.diag(h) + Qp
        step = np.linalg.solve(H, grad)
        theta = theta + step
        if np.abs(step).max() < 1e-10:
            break
    z = z_sign * (theta[z_years] - opp)
    lam = _lam(z)
    h = np.zeros(T)
    np.add.at(h, z_years, lam * (z + lam))
    H = np.diag(h) + Qp
    ll = log_ndtr(z).sum()
    dev = theta - m
    sign_p, logdet_Qp = np.linalg.slogdet(Qp)
    sign_h, logdet_H = np.linalg.slogdet(H)
    log_evidence = (ll + 0.5 * logdet_Qp - 0.5 * dev @ Qp @ dev
                    - 0.5 * logdet_H)
    return theta, H, log_evidence


def main(out_path, first_year=1985, last_year=2024):
    winners, losers, years = load(first_year, last_year)
    print(f"{len(winners)} matches, {first_year}-{last_year}")
    names, idx, a, keep, wi, li = static_abilities(winners, losers)
    print(f"{len(names)} players in the static fit; "
          f"ability sd {a.std():.3f}")
    ykeep = years[keep]

    out = {"first_year": first_year, "last_year": last_year,
           "n_matches": len(keep), "n_players": len(names), "players": []}
    for player in PLAYERS:
        if player not in idx:
            print(f"  (skipping {player}: not in the fit)")
            continue
        p = idx[player]
        sel = (wi == p) | (li == p)
        if sel.sum() < 100:
            continue
        yrs = ykeep[sel]
        y0, y1 = yrs.min(), yrs.max()
        z_years = yrs - y0
        won = (wi[sel] == p)
        z_sign = np.where(won, 1.0, -1.0)
        opp = np.where(won, a[li[sel]], a[wi[sel]])

        best = None
        for tau in np.arange(0.05, 0.55, 0.05):
            th, H, ev = path_posterior(z_years, z_sign, opp, a[p], tau)
            if best is None or ev > best[3]:
                best = (th, H, tau, ev)
        theta, H, tau, ev = best
        T = len(theta)
        per_year = np.bincount(z_years, minlength=T).tolist()
        out["players"].append({
            "name": player, "first": int(y0), "last": int(y1),
            "tau": float(tau), "log_evidence": float(ev),
            "matches": int(sel.sum()), "per_year": per_year,
            "mean": theta.tolist(),
            "Q_diag": np.diag(H).tolist(),
            "Q_off": np.diag(H, 1).tolist(),
        })
        peak = int(np.argmax(theta))
        print(f"  {player:18s} {y0}-{y1}  tau {tau:.2f}  "
              f"MAP peak {y0 + peak}  ({sel.sum()} matches)")

    json.dump(out, open(out_path, "w"))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1],
         int(sys.argv[2]) if len(sys.argv) > 2 else 1985,
         int(sys.argv[3]) if len(sys.argv) > 3 else 2024)
