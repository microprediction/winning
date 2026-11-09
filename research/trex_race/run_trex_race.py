"""T-Rex variable selection read as a race against reference runners (#1030).

Machkour, Muma and Palomar (Signal Processing 231, 2025): K random
experiments, each appending L standard-normal dummies to the p candidates
and running LARS until T dummies have entered; a candidate's relative
occurrence Phi_{T,L}(j) is the fraction of experiments in which it entered
before termination.

The race reading. At any step a candidate's performance is the absolute
correlation of its column with the current residual. The dummies are
i.i.d. and redrawn in every experiment, so in each experiment the dummy
field is L independent half-normal runners (|N(0, 1)| in units of the
dummy sampling sd). A REAL column's statistic is fixed by the data and
does not change between experiments (only slightly, through the T - 1
dummies already in the model), so a real variable is a runner with a
fixed performance s and NO fresh noise: T-Rex randomises the field, not
the runner. Then

    P_T(s) = P(fewer than T of the L dummies beat a fixed s)
           = P(Binomial(L, P(|Z| > s)) <= T - 1)

is the top-k membership probability of a noiseless runner in an identical
half-normal field. Given the data the K inclusion indicators are i.i.d.
Bernoulli, so Phi_{T,L}(j) ~ Binomial(K, q_j) / K exactly; the race model
is the approximation q_j = P_T(s_j): a candidate's statistic is its
correlation with the CURRENT residual, which moves with the variables and
the up-to-T-1 dummies already on the path, and LARS enters by that, not
by the marginal correlation. The nulls are
exchangeable with the dummies, so their strengths are s ~ |N(0, 1)| and
an active's is large. Inverting Phi through P_T gives a strength on the
dummy scale; integrating the selection event over the null law gives a
model-based count of selected nulls, the twin of the paper's deflated
V-hat. The identical field makes P_T a closed form, used here; the
general engine (winning.factor.topk.top_k_probabilities with the runner's
variance D_j small) is checked against it and converges to the noiseless
limit as D_j -> 0 (recorded in the results).

Setup (their Fig. 5/6): n = 300, p = 1000, p1 = 10, SNR = 1, K = 20,
L = p. Everything here is pure numpy + sklearn's LARS + winning's top-k.
sklearn is not in CI; this is research code, not a test.

Usage: OMP_NUM_THREADS=1 python run_trex_race.py [MC] [L_MULT]
"""
import json, os, sys, time
import numpy as np
from scipy.stats import binom, spearmanr, norm
from sklearn.linear_model import lars_path

from winning.factor.topk import top_k_probabilities

HERE = os.path.dirname(os.path.abspath(__file__))
N, P, P1, SNR, K = 300, 1000, 10, 1.0, 20
MC = int(sys.argv[1]) if (len(sys.argv) > 1 and __name__ == "__main__") else 50
L_MULT = int(sys.argv[2]) if (len(sys.argv) > 2 and __name__ == "__main__") else 1
L = L_MULT * P
T_CAP = int(os.environ.get("T_CAP", 50))   # the calibration may run T this far; the grid shows 1..20
ALPHA = 0.10
V_GRID = np.round(np.arange(0.5, 1.0, 1.0 / K), 3)      # 0.5, 0.55, ..., 0.95 (Delta v = 1/K)
T_GRID = np.arange(1, min(20, T_CAP) + 1)
S_GRID = np.linspace(0.0, 9.0, 181)                     # strengths on the dummy scale (absolute correlations)
MAX_ITER = N - 2


# ---------------------------------------------------------------- the race map
def race_table(L, T_cap, s_grid):
    """P_T(s) for T = 1..T_cap on s_grid: a noiseless runner of strength s
    is among the first T of itself and L i.i.d. half-normal dummies iff
    fewer than T dummies exceed s. Closed form on the identical field."""
    p_beat = 2.0 * norm.sf(np.asarray(s_grid, float))
    return np.vstack([binom.cdf(T - 1, L, p_beat) for T in range(1, T_cap + 1)])


def engine_check(L, T, s, dj, points):
    """The general engine on the same race: the runner at mu = -s with a
    small idiosyncratic variance dj, the dummies at 0 with unit variance
    (a normal, not half-normal, field; for s >= 0 the two fields are
    compared at the same tail mass by placing the dummies at the normal
    quantile of the half-normal survival). Returns the engine's value and
    the closed form it should approach as dj -> 0."""
    mu = np.zeros(L + 1); mu[0] = -norm.isf(2.0 * norm.sf(s))        # same P(dummy beats s) under a normal field
    D = np.ones(L + 1); D[0] = dj
    q = float(top_k_probabilities(mu, T, D=D, points=points)[0])
    return q, float(binom.cdf(T - 1, L, 2.0 * norm.sf(s)))


def invert(phi, row, s_grid):
    """Strength from a relative occurrence by monotone interpolation of
    the race map (row = P_T on s_grid, increasing in s)."""
    return np.interp(phi, row, s_grid)


# ---------------------------------------------------------------- one data set
def simulate(rng, graded=False):
    X = rng.standard_normal((N, P))
    beta = np.zeros(P)
    act = rng.choice(P, P1, replace=False)
    beta[act] = np.repeat([0.25, 0.5, 0.75, 1.0, 1.25], 2) if graded else 1.0
    signal = X @ beta
    sigma = np.sqrt(signal.var() / SNR)
    y = signal + sigma * rng.standard_normal(N)
    return X, y, beta, act


def experiments(rng, X, y, L, K, T_cap):
    """Phi_{t,L}(j) for t = 1..T_cap, shape (T_cap, P). Each experiment is
    one LARS path on [X, dummies]; candidate j is included at termination
    T iff it entered before the T-th dummy."""
    inc = np.zeros((K, T_cap, P), dtype=bool)
    n_dummies_seen = []
    for k in range(K):
        D = rng.standard_normal((N, L))
        _, active, _ = lars_path(np.hstack([X, D]), y, method="lar", max_iter=MAX_ITER)
        active = np.asarray(active)
        pos = {int(j): i for i, j in enumerate(active)}
        dummy_pos = np.sort([i for i, j in enumerate(active) if j >= P])
        n_dummies_seen.append(len(dummy_pos))
        cand_pos = np.full(P, np.inf)
        for j, i in pos.items():
            if j < P:
                cand_pos[j] = i
        for t in range(1, T_cap + 1):
            if t <= len(dummy_pos):
                inc[k, t - 1] = cand_pos < dummy_pos[t - 1]
            else:
                inc[k, t - 1] = cand_pos < np.inf          # path ended before T dummies: all entrants
    return inc.mean(axis=0), int(min(n_dummies_seen))


# ---------------------------------------------------------------- T-Rex estimators
def deflated(phi, L, eps=2.2e-16):
    """Definition 5 as the R package computes it (TRexSelector::Phi_prime_fun):
    for the stopping point T, the set A = {Phi_T > 0.5} is fixed, and the
    step-t increments are scaled by
        1 - [(p - sum_q Phi_t(q)) / (L - t + 1)] / sum_{q in A} dPhi_t(q)
    (0 when the denominator vanishes). phi: (T_cap, P); returns (T_cap, P),
    row T - 1 being Phi'_{T,L}."""
    T_cap, p = phi.shape
    dphi = np.diff(np.vstack([np.zeros(p), phi]), axis=0)
    av = phi.sum(axis=1)                                         # sum_q Phi_t(q)
    out = np.zeros_like(phi)
    for T in range(1, T_cap + 1):
        A = phi[T - 1] > 0.5
        dA = dphi[:T][:, A].sum(axis=1)                          # sum over A of the increments, by step
        scale = np.zeros(T)
        for t in range(1, T + 1):
            if dA[t - 1] > eps:
                scale[t - 1] = 1.0 - ((p - av[t - 1]) / (L - t + 1)) / dA[t - 1]
        out[T - 1] = scale @ dphi[:T]
    return out


def fdp_hat(phi_T, phi_def_T, v):
    sel = phi_T > v
    R = sel.sum()
    V = (1.0 - phi_def_T[sel]).sum()
    return min(1.0, V / R) if R else 0.0, R, V


def model_selected_nulls(T, v, tab, s_grid, K, p0):
    """Race-model expected number of selected nulls: a null of strength s
    (s ~ |N(0,1)|, the dummy law) is selected iff Binomial(K, P_T(s)) > vK."""
    # the selection event is a steep function of s; integrate on the
    # strength grid itself against the half-normal density
    w = 2.0 * norm.pdf(s_grid); w = w / w.sum()
    pT = tab[T - 1]
    m = int(np.floor(v * K)) + 1                       # strict: Phi > v
    return p0 * float(np.sum(w * binom.sf(m - 1, K, pT)))


def calibrate(fdp_fn, T_max, alpha):
    """Algorithm 1: raise T while FDP(v = 1 - Delta v, T) <= alpha, then
    the (v, T') with the most selections among T' < T and all v."""
    best = (0, None, None)
    T = 1
    while T <= T_max and fdp_fn(T, V_GRID[-1])[0] <= alpha:
        for v in V_GRID:
            f, R = fdp_fn(T, v)[:2]
            if f <= alpha and R >= best[0] and R > 0:          # ties: the largest v, as the R package
                best = (R, float(v), T)
        T += 1
    return best


def write_results(obj, stem, path=None):
    """Atomic write next to the script; an existing file is never
    overwritten: the new one takes a UTC time stamp (as run_f1_season does).
    With path= given (a run's own file), it is rewritten in place, atomically."""
    if path is None:
        path = os.path.join(HERE, stem + ".json")
        if os.path.exists(path):
            stamp = time.strftime('%Y%m%dT%H%M%S', time.gmtime())
            path = os.path.join(HERE, f"{stem}_{stamp}_{os.getpid()}.json")
        # claim the name exclusively: a second writer in the same second, or
        # another process, gets an error instead of replacing this file
        os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644))
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    rng = np.random.default_rng(1030)
    print(f"race table: L={L}, T<= {T_CAP}, {len(S_GRID)} strengths", flush=True)
    tab = race_table(L, T_CAP, S_GRID)
    ref = []
    for T, s in ((1, 3.0), (5, 3.0), (20, 2.0)):
        if T > T_CAP:
            continue
        for dj, pts in ((0.04, 513), (0.01, 1025)):
            q, exact = engine_check(L, T, s, dj, pts)
            ref.append({"T": T, "s": s, "D_j": dj, "points": pts, "engine": q, "closed_form": exact})
    print("  engine check", ref, f"{time.time() - t0:.0f}s", flush=True)

    # one data set, in detail: strengths, their spread among nulls, the Binomial law of Phi
    X, y, beta, act = simulate(rng)
    phi, dmin = experiments(rng, X, y, L, K, T_CAP)
    marg_t = np.sqrt(N) * np.abs(X.T @ y) / (np.linalg.norm(y) * np.sqrt(N))      # marginal |corr| * sqrt(n)
    detail = {}
    for T in [t for t in (1, 5, 10, 20) if t <= T_CAP]:
        phi_s = (K * phi[T - 1] + 0.5) / (K + 1)          # Phi in {0, 1/K, .., 1}; 0 and 1 invert to -+inf
        s_hat = invert(phi_s, tab[T - 1], S_GRID)
        null = np.ones(P, bool); null[act] = False
        # the race law for nulls: Phi_T ~ Bin(K, P_T(s)) / K with s ~ N(0, 1)
        pred = {str(v): model_selected_nulls(T, v, tab, S_GRID, K, int(null.sum())) for v in (0.0, 0.25, 0.5, 0.75)}
        emp = {str(v): int((phi[T - 1][null] > v).sum()) for v in (0.0, 0.25, 0.5, 0.75)}
        detail[int(T)] = {
            "phi_actives": phi[T - 1][act].round(3).tolist(),
            "strength_actives": s_hat[act].round(2).tolist(),
            "marginal_t_actives": marg_t[act].round(2).tolist(),
            "null_strength_mean_sd": [float(s_hat[null].mean()), float(s_hat[null].std())],
            "null_strength_clipped_at_floor": int((phi[T - 1][null] == 0).sum()),
            "spearman_strength_vs_marginal_t_all": float(spearmanr(s_hat, marg_t).correlation),
            "spearman_strength_vs_marginal_t_nulls": float(spearmanr(s_hat[null], marg_t[null]).correlation),
            "spearman_strength_vs_beta_all": float(spearmanr(s_hat, np.abs(beta)).correlation),
            "nulls_with_phi_above_v": {"predicted": pred, "observed": emp},
        }
    # graded betas: calibration of strength against the truth on the dummy scale
    Xg, yg, betag, actg = simulate(rng, graded=True)
    phig, _ = experiments(rng, Xg, yg, L, K, T_CAP)
    graded = {}
    for T in [t for t in (5, 10) if t <= T_CAP]:
        s_hat = invert((K * phig[T - 1] + 0.5) / (K + 1), tab[T - 1], S_GRID)
        mt = np.sqrt(N) * np.abs(Xg.T @ yg) / (np.linalg.norm(yg) * np.sqrt(N))
        graded[int(T)] = {"beta": betag[actg].tolist(), "phi": phig[T - 1][actg].round(3).tolist(),
                          "strength": s_hat[actg].round(2).tolist(), "marginal_t": mt[actg].round(2).tolist(),
                          "spearman_strength_vs_beta_actives": float(spearmanr(s_hat[actg], betag[actg]).correlation),
                          "spearman_marginal_t_vs_beta_actives": float(spearmanr(mt[actg], betag[actg]).correlation)}

    # Monte Carlo: the Fig. 5 grid and the calibrated points. Every completed
    # repetition is appended to a JSONL log as it finishes (one line each,
    # flushed), and the summary is built from that log, so an interruption
    # keeps what was done.
    log_path = os.path.join(HERE, f"log_L{L_MULT}p_MC{MC}_{time.strftime('%Y%m%dT%H%M%S', time.gmtime(t0))}.jsonl")
    log = open(log_path, "a")
    for rep in range(MC):
        X, y, beta, act = simulate(rng)
        phi, dmin = experiments(rng, X, y, L, K, T_CAP)
        phi_def = deflated(phi, L)
        is_act = np.zeros(P, bool); is_act[act] = True
        g = {k_: np.zeros((len(T_GRID), len(V_GRID))) for k_ in ("fdp", "fdp_hat", "fdp_model", "tpp")}
        for ti, T in enumerate(T_GRID):
            for vi, v in enumerate(V_GRID):
                f, R, V = fdp_hat(phi[T - 1], phi_def[T - 1], v)
                sel = phi[T - 1] > v
                g["fdp"][ti, vi] = (sel & ~is_act).sum() / max(R, 1)
                g["fdp_hat"][ti, vi] = f
                g["fdp_model"][ti, vi] = model_selected_nulls(T, v, tab, S_GRID, K, P - P1) / max(R, 1)   # p0 = p - p1 in simulation
                g["tpp"][ti, vi] = (sel & is_act).sum() / P1
        if dmin < 1:
            raise RuntimeError("an experiment's path ended before any dummy entered: raise MAX_ITER")
        T_max = min(T_CAP, dmin)                       # >= 1: Algorithm 1 needs T = 1 at least
        cal_row = {}
        for name, fn in (("trex", lambda T, v: fdp_hat(phi[T - 1], phi_def[T - 1], v)),
                         ("model", lambda T, v: (model_selected_nulls(T, v, tab, S_GRID, K, P - P1) / max((phi[T - 1] > v).sum(), 1),
                                                 int((phi[T - 1] > v).sum())))):
            R, v, T = calibrate(fn, T_max, ALPHA)
            if R == 0:
                cal_row[name] = {"R": 0, "fdp": 0.0, "tpp": 0.0, "v": None, "T": None}
            else:
                sel = phi[T - 1] > v
                cal_row[name] = {"R": int(R), "fdp": float((sel & ~is_act).sum() / R), "tpp": float((sel & is_act).sum() / P1),
                                 "v": v, "T": int(T)}
        log.write(json.dumps({"rep": rep, "dmin": int(dmin), "T_max": int(T_max),
                              "grid": {k_: g[k_].round(6).tolist() for k_ in g}, "cal": cal_row}) + "\n"); log.flush()
        if (rep + 1) % 10 == 0:
            print(f"  rep {rep + 1}/{MC} {time.time() - t0:.0f}s", flush=True)
    log.close()
    rows = [json.loads(line) for line in open(log_path)]
    grid = {k_: np.mean([r["grid"][k_] for r in rows], axis=0).round(4).tolist() for k_ in ("fdp", "fdp_hat", "fdp_model", "tpp")}
    cal = {name: [r["cal"][name] for r in rows] for name in ("trex", "model")}

    def summ(rows):
        a = np.array([[r["fdp"], r["tpp"], r["R"]] for r in rows])
        return {"fdr": float(a[:, 0].mean()), "fdr_se": float(a[:, 0].std(ddof=1) / np.sqrt(len(a))),
                "tpr": float(a[:, 1].mean()), "tpr_se": float(a[:, 1].std(ddof=1) / np.sqrt(len(a))),
                "mean_selected": float(a[:, 2].mean()),
                "T_star_median": float(np.median([r["T"] for r in rows if r["T"]])) if any(r["T"] for r in rows) else None,
                "v_star_median": float(np.median([r["v"] for r in rows if r["v"]])) if any(r["v"] for r in rows) else None}
    out = {"setup": dict(n=N, p=P, p1=P1, snr=SNR, K=K, L=L, T_cap=T_CAP, alpha=ALPHA, MC=len(rows), s_grid=[float(S_GRID[0]), float(S_GRID[-1]), len(S_GRID)]),
           "log": os.path.basename(log_path), "dmin_over_reps": [int(min(r["dmin"] for r in rows)), int(max(r["dmin"] for r in rows))],
           "engine_check": ref, "detail_one_dataset": detail, "graded_betas": graded,
           "grid": {"v": V_GRID.tolist(), "T": T_GRID.tolist(), **grid},
           "calibrated": {k_: summ(v_) for k_, v_ in cal.items()}, "seconds": time.time() - t0}
    path = write_results(out, f"results_L{L_MULT}p_MC{MC}")
    print("->", path)
    print(json.dumps(out["calibrated"], indent=1))


if __name__ == "__main__":
    main()
