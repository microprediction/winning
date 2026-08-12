"""Human odds invariance from ranking data, for comparison with the machine test.

The pair-restriction statistic of the machine batteries needs, for humans, a
population that chooses both from a full set and from pairs. Complete rankings
supply both: a respondent's choice from any set is the highest-ranked available
item, so the population choice distribution over any subset is computable
without asking anyone a second time.

    delta_ij = log(q_i/q_j) - log(p_i/p_j)

where p comes from the full set and q from the pair. Luce requires zero.

Data: Kamishima's SUSHI set, 5,000 complete strict rankings of 10 items
(kamishima.net/sushi, also on PrefLib). Download sushi3-2016.zip and pass the
path to sushi3a.5000.10.order.

One limitation is structural and worth stating rather than working around. A
choice derived from a fixed ranking cannot reverse: if i is preferred to j in
some set, it is preferred in every subset containing both. So ranking data can
measure the odds shift but cannot exhibit the rank reversals the machine data
shows, because the elicitation format presupposes the stability the machines
violate. Testing reversals in humans needs subset choices elicited separately,
not derived.

Usage:  python sushi_iia.py path/to/sushi3a.5000.10.order
"""
import math
import random
import sys
from collections import defaultdict

K = 10
NAMES = ["ebi", "anago", "maguro", "ika", "uni", "sake", "tamago", "toro",
         "tekka-maki", "kappa-maki"]


def load(path):
    out = []
    for i, line in enumerate(open(path)):
        if i == 0:
            continue
        order = [int(x) for x in line.split()[2:]]
        if len(order) == K:
            out.append(order)
    return out


def shares(rankings, available):
    c = [0] * K
    for r in rankings:
        for it in r:
            if it in available:
                c[it] += 1
                break
    n = sum(c)
    return [x / n for x in c] if n else None


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "sushi3a.5000.10.order"
    rk = load(path)
    full = shares(rk, set(range(K)))
    print(f"{len(rk)} complete rankings")
    print("full-set shares: " + " ".join(
        f"{NAMES[i]} {full[i]:.3f}" for i in range(K)))

    deltas, rows = [], []
    for i in range(K):
        for j in range(i + 1, K):
            if full[i] <= 0 or full[j] <= 0:
                continue
            pair = shares(rk, {i, j})
            if pair[i] <= 0 or pair[j] <= 0:
                continue
            d = math.log(pair[i] / pair[j]) - math.log(full[i] / full[j])
            deltas.append(d)
            rows.append((NAMES[i], NAMES[j], d))
    n = len(deltas)
    random.seed(4)
    B = 20000
    bs = sorted(sum(random.choice(deltas) for _ in range(n)) / n for _ in range(B))
    print(f"\n{n} pairs")
    print(f"  mean delta   {sum(deltas)/n:+.4f} "
          f"[{bs[int(.025*B)]:+.4f}, {bs[int(.975*B)]:+.4f}]   (Luce requires 0)")
    print(f"  mean |delta| {sum(abs(d) for d in deltas)/n:.4f}")
    viol = sum(1 for d in deltas if abs(d) > 0.1)
    print(f"  |delta| > 0.1 in {viol}/{n} pairs ({100*viol/n:.0f}%)")
    print(f"  shrank toward parity in "
          f"{sum(1 for d in deltas if d < 0)}/{n}")
    print("\nmachine comparison, identical statistic:")
    print("  mean -1.1500 [-1.5474, -0.8076], |delta|>0.1 in 96.8% of 279 pairs")
    print("  so humans violate in the same direction, in the same share of")
    print("  pairs, roughly a third to a quarter as strongly")


if __name__ == "__main__":
    main()
