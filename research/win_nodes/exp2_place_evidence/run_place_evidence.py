"""How much each kind of outcome evidence tells a win node.

One factor-correlated race, eight entrants, one shared factor, prior
N(0, I) on the skills. After the race we learn one of: nothing; that a
named entrant finished in the top three; the top three as a set, order
unknown; the order among three named entrants (saying nothing about the
rest); the top three in order with the rest behind; the winner only; the
full finishing order. Each is folded in by its exact message
(update_topk_correlated, update_topset_correlated, update_prefix_correlated,
update_order_correlated, update_winner_correlated). Posterior error of
the skill contrasts against the truth and the log score on the winner
of an independent rerun, averaged over seeded races.

Who is named must not leak the outcome beyond what the message says
(review of #1024): the entrant reported in the top three is drawn
uniformly from the actual top three, so the message carries exactly
"this one is in the first three" and nothing about their rank within
it; the three entrants whose mutual order is reported are a uniform
random subset of the field, not the actual first three. Naming the
winner, or the first three, would smuggle the stronger message in.

Reproduce: OMP_NUM_THREADS=1 python research/win_nodes/exp2_place_evidence/run_place_evidence.py
"""
import json, os, time
import numpy as np

from winning.factor.races import race_probabilities
from winning.ratings.nway import (update_winner_correlated, update_order_correlated,
                                  update_topk_correlated, update_topset_correlated,
                                  update_prefix_correlated)

N, RANK, BETA2, SEEDS, K = 8, 1, 1.0, 1600, 3
HERE = os.path.dirname(os.path.abspath(__file__))
KINDS = ["prior", "member of top 3", "order among 3 named", "top-3 set", "winner", "top 3 in order, rest behind", "full order"]
# Who is named (see the docstring). Recorded in every raw row and in the
# summary so rows from a run under another protocol cannot be mixed in.
PROTOCOL = ("named: uniform member of the actual first K; "
            "trio: uniform K-subset of the field, reported in finishing order")


def centred(x):
    x = np.asarray(x, float)
    return x - x.mean()


def write_json_atomic(path, obj):
    """Sibling temp file then rename: a crash mid-write leaves no truncated file."""
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def main():
    rng = np.random.default_rng(20261108)
    err = {k: [] for k in KINDS}; var = {k: [] for k in KINDS}; score = {k: 0.0 for k in KINDS}
    t0 = time.time()
    run_id = f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime(t0))}-{os.getpid()}"
    # raw per-race results go to this run's own append-only log as they are
    # produced (created exclusively: one file per run, so a rerun can never
    # append rows with repeated seeds to an earlier run's log); an interrupted
    # run loses nothing and paired differences can be read back
    log_path = os.path.join(HERE, f"races_{run_id}.jsonl")
    log = open(log_path, "x")
    for seed in range(SEEDS):
        s = rng.normal(size=N)
        V = rng.normal(scale=0.6, size=(N, RANK))
        D = np.full(N, BETA2)
        m0, v0 = np.zeros(N), np.ones(N)
        f = rng.normal(size=RANK); eps = rng.normal(size=N)
        order = list(np.argsort(-(s + V @ f + np.sqrt(D) * eps)))      # max wins
        w = order[0]
        named = int(rng.choice(order[:K]))                                # a uniform member of the first K
        trio = sorted(rng.choice(N, size=K, replace=False), key=order.index)   # a uniform K-subset, in finishing order
        posts = {
            "prior": (m0, v0),
            "member of top 3": update_topk_correlated(m0, v0, named, K, V=V, beta2=BETA2)[:2],
            "top-3 set": update_topset_correlated(m0, v0, order[:K], V=V, beta2=BETA2)[:2],
            "winner": update_winner_correlated(m0, v0, w, V, beta2=BETA2)[:2],
            "order among 3 named": update_order_correlated(m0, v0, trio, V, beta2=BETA2)[:2],
            "top 3 in order, rest behind": update_prefix_correlated(m0, v0, order[:K], V=V, beta2=BETA2)[:2],
            "full order": update_order_correlated(m0, v0, order, V, beta2=BETA2)[:2],
        }
        f2 = rng.normal(size=RANK); eps2 = rng.normal(size=N)
        w2 = int(np.argmax(s + V @ f2 + np.sqrt(D) * eps2))
        row = {"run_id": run_id, "protocol": PROTOCOL, "seed": seed}
        for k, (m, v) in posts.items():
            e = float(np.sqrt(np.mean((centred(m) - centred(s)) ** 2))); vv = float(np.mean(v))
            pred = np.asarray(race_probabilities(-np.asarray(m), V=V, D=D + np.asarray(v)))
            ls = float(np.log(pred[w2]))
            err[k].append(e); var[k].append(vv); score[k] += ls
            row[k] = {"rmse": e, "var": vv, "log_score": ls}
        log.write(json.dumps(row) + "\n"); log.flush()
    log.close()
    se = {k: float(np.std(err[k], ddof=1) / np.sqrt(SEEDS)) for k in KINDS}
    out = {"n": N, "rank": RANK, "beta2": BETA2, "seeds": SEEDS, "k": K,
           "rmse_contrast": {k: float(np.mean(err[k])) for k in KINDS},
           "mean_posterior_var": {k: float(np.mean(var[k])) for k in KINDS},
           "mean_log_score_next_winner": {k: score[k] / SEEDS for k in KINDS},
           "se_rmse_contrast": se,
           "paired_rmse_diff_vs_prior_se": {k: float(np.std(np.asarray(err[k]) - np.asarray(err["prior"]), ddof=1) / np.sqrt(SEEDS))
                                           for k in KINDS if k != "prior"},
           "run_id": run_id, "protocol": PROTOCOL,
           "raw_log": os.path.basename(log_path), "seconds": time.time() - t0}
    # never overwrite an existing results.json silently: a rerun gets a stamped name
    path = os.path.join(HERE, "results.json")
    if os.path.exists(path):
        path = os.path.join(HERE, f"results_{run_id}.json")
    write_json_atomic(path, out)
    print("summary ->", path, " raw ->", log_path)
    for k in KINDS:
        print(f"{k:28s} rmse {out['rmse_contrast'][k]:.4f}  var {out['mean_posterior_var'][k]:.4f}"
              f"  log score {out['mean_log_score_next_winner'][k]:.4f}")
    print(f"{out['seconds']:.1f}s")


if __name__ == "__main__":
    main()
