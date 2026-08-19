"""Phase 3: analysis.

For each word: generated sense distribution (from judge labels), stated
distribution, normalized entropies. Then fit two one-parameter maps from
stated -> generated across all words:

  Luce-gamma:    p_i ~ s_i^gamma            (power tilt / temperature)
  Thurstone-sig: p_i = P(X_i wins), X_i ~ N(a_i, sigma), where the locations
                 a_i are calibrated so that unit-noise win probs equal s_i.

Smaller RMSE wins. Pure Python (no numpy).
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).parent

SQRT2 = math.sqrt(2.0)


def phi(x):  # standard normal pdf
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def Phi(x):  # standard normal cdf
    return 0.5 * (1.0 + math.erf(x / SQRT2))


def win_probs(a):
    """P(X_i = max), X_i ~ N(a_i, 1), by numerical integration."""
    lo = min(a) - 8.0
    hi = max(a) + 8.0
    step = 0.02
    n = len(a)
    probs = [0.0] * n
    x = lo
    while x <= hi:
        cdfs = [Phi(x - aj) for aj in a]
        for i in range(n):
            others = 1.0
            for j in range(n):
                if j != i:
                    others *= cdfs[j]
            probs[i] += phi(x - a[i]) * others
        x += step
    total = sum(probs)
    return [p * step / (total * step) for p in probs]  # renormalize


def calibrate_locations(target, iters=300, lr=0.8):
    """Find locations a (unit noise) whose win probs match target."""
    n = len(target)
    a = [0.0] * n
    for _ in range(iters):
        p = win_probs(a)
        a = [ai + lr * (math.log(max(t, 1e-9)) - math.log(max(pi, 1e-9)))
             for ai, t, pi in zip(a, target, p)]
        mean = sum(a) / n
        a = [ai - mean for ai in a]
    return a


def luce_map(s, gamma):
    w = [max(x, 1e-9) ** gamma for x in s]
    z = sum(w)
    return [x / z for x in w]


def thurstone_map(a, sigma):
    return win_probs([ai / sigma for ai in a])


def rmse(pairs):
    return math.sqrt(sum((p - q) ** 2 for p, q in pairs) / len(pairs))


def entropy_norm(p):
    n = len(p)
    if n < 2:
        return 0.0
    h = -sum(x * math.log2(x) for x in p if x > 0)
    return h / math.log2(n)


def fit_param(words_data, family, grid):
    best = None
    for theta in grid:
        pairs = []
        for wd in words_data:
            pred = (luce_map(wd["stated"], theta) if family == "luce"
                    else thurstone_map(wd["a"], theta))
            pairs += list(zip(pred, wd["generated"]))
        err = rmse(pairs)
        if best is None or err < best[1]:
            best = (theta, err)
    return best


def main():
    stimuli = json.loads((HERE / "stimuli.json").read_text())
    judged = json.loads((HERE / "judged.json").read_text())
    stated_raw = json.loads((HERE / "stated.json").read_text())

    words_data = []
    print(f"{'word':<8} {'H_gen':>6} {'H_stated':>9}  generated vs stated")
    for word, senses in stimuli.items():
        if word not in judged or word not in stated_raw:
            continue
        labels = judged[word]
        counts = {s: 0 for s in senses}
        fallback = 0
        for lab in labels:
            key = str(lab).strip().lower()
            if key in counts:
                counts[key] += 1
            else:
                fallback += 1
        tot = sum(counts.values())
        if tot == 0:
            continue
        generated = [counts[s] / tot for s in senses]

        sr = {str(k).strip().lower(): float(v) for k, v in stated_raw[word].items()}
        stated = [max(sr.get(s, 0.0), 0.0) for s in senses]
        z = sum(stated)
        if z <= 0:
            continue
        stated = [x / z for x in stated]

        a = calibrate_locations(stated)
        words_data.append({"word": word, "senses": senses,
                           "generated": generated, "stated": stated,
                           "a": a, "fallback": fallback})
        print(f"{word:<8} {entropy_norm(generated):>6.3f} {entropy_norm(stated):>9.3f}  "
              + ", ".join(f"{s}:{g:.2f}/{st:.2f}" for s, g, st in zip(senses, generated, stated)))

    mean_h_gen = sum(entropy_norm(w["generated"]) for w in words_data) / len(words_data)
    mean_h_stated = sum(entropy_norm(w["stated"]) for w in words_data) / len(words_data)
    print(f"\nmean normalized entropy: generated={mean_h_gen:.3f}  stated={mean_h_stated:.3f}")
    total_fb = sum(w["fallback"] for w in words_data)
    print(f"fallback labels (multiple/unclear/other): {total_fb}")

    lgrid = [0.25 * i for i in range(1, 81)]                       # gamma 0.25..20
    sgrid = [math.exp(math.log(0.05) + i * (math.log(3.0) - math.log(0.05)) / 79)
             for i in range(80)]                                    # sigma 0.05..3
    g_best, g_err = fit_param(words_data, "luce", lgrid)
    s_best, s_err = fit_param(words_data, "thurstone", sgrid)

    # Baseline: identity (stated as-is predicts generated)
    id_pairs = [pq for wd in words_data for pq in zip(wd["stated"], wd["generated"])]
    print(f"\nidentity baseline RMSE: {rmse(id_pairs):.4f}")
    print(f"Luce power-tilt:  gamma={g_best:.2f}  RMSE={g_err:.4f}")
    print(f"Thurstone:        sigma={s_best:.3f}  RMSE={s_err:.4f}")

    # Per-word winner tally at the globally fitted parameters
    luce_wins = thur_wins = 0
    for wd in words_data:
        lp = rmse(list(zip(luce_map(wd["stated"], g_best), wd["generated"])))
        tp = rmse(list(zip(thurstone_map(wd["a"], s_best), wd["generated"])))
        if tp < lp:
            thur_wins += 1
        else:
            luce_wins += 1
    print(f"per-word winners: Thurstone {thur_wins}, Luce {luce_wins}")

    (HERE / "analysis.json").write_text(json.dumps({
        "mean_H_generated": mean_h_gen, "mean_H_stated": mean_h_stated,
        "luce": {"gamma": g_best, "rmse": g_err},
        "thurstone": {"sigma": s_best, "rmse": s_err},
        "words": [{k: v for k, v in wd.items() if k != "a"} for wd in words_data],
    }, indent=1))


if __name__ == "__main__":
    main()
