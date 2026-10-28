"""Compile runs/sweep_*.csv into the markdown tables in RESULTS.md.

    python research/factor_ghk/compile_results.py > /tmp/tables.md
"""
import csv, glob, os

HERE = os.path.dirname(os.path.abspath(__file__))
rows = []
for f in sorted(glob.glob(os.path.join(HERE, "runs", "sweep_*.csv"))):
    rows += list(csv.DictReader(open(f)))
fields = sorted({(r["rank"], r["D"], r["margin"]) for r in rows}, key=lambda k: (int(k[0]), -float(k[1]), float(k[2])))
by = {(r["rank"], r["D"], r["margin"], r["method"]): r for r in rows}
first = {k: next(r for r in rows if (r["rank"], r["D"], r["margin"]) == k) for k in fields}

def col(k):
    return f"r{k[0]} D{k[1]}" + ("" if k[2] == "4.0" else f" m{k[2][0]}")

print("### Fields and truth certification\n")
print("| field | sharpness | truth nodes | truth vs 2nd scramble | truth vs MC 3M (se ~3e-4) | hybrid passes at R=2048 |")
print("|---|---:|---:|---:|---:|---:|")
for k in fields:
    r = first[k]; a = by[(k[0], k[1], k[2], "Hybrid A per-winner R=2048 (normalised)")]
    print(f"| {col(k)} | {float(r['sharp']):.1f} | 2^{r['truth_m']} | {float(r['truth_vs_scramble']):.1e} | {float(r['truth_vs_mc']):.1e} | {a['passes']} |")

want = [("winning default rule", "default rule (= Sobol 2^13 at rank >= 2)"),
        ("fixed scrambled Sobol 2^7", "Sobol 2^7 (128 passes)"), ("fixed scrambled Sobol 2^9", "Sobol 2^9 (512)"),
        ("fixed scrambled Sobol 2^11", "Sobol 2^11 (2048)"), ("fixed scrambled Sobol 2^13", "Sobol 2^13 (8192)"),
        ("Hybrid A per-winner R=32 (normalised)", "Hybrid A per-winner R=32 (~256 passes)"),
        ("Hybrid A per-winner R=128 (normalised)", "Hybrid A per-winner R=128 (~1024)"),
        ("Hybrid A per-winner R=512 (normalised)", "Hybrid A per-winner R=512 (~4096)"),
        ("Hybrid A per-winner R=2048 (normalised)", "Hybrid A per-winner R=2048 (~16384)"),
        ("Hybrid A shared R=256 (normalised)", "Hybrid A shared R=256 (256 passes)"),
        ("Hybrid A shared R=1024 (normalised)", "Hybrid A shared R=1024 (1024)"),
        ("Hybrid A shared R=4096 (normalised)", "Hybrid A shared R=4096 (4096)"),
        ("Hybrid A shared R=16384 (normalised)", "Hybrid A shared R=16384 (16384)"),
        ("qmc_ghk (runner contrasts) B=512 (normalised)", "qmc_ghk B=512 (no lattice)"),
        ("qmc_ghk (runner contrasts) B=2048 (normalised)", "qmc_ghk B=2048"),
        ("qmc_ghk (runner contrasts) B=8192 (normalised)", "qmc_ghk B=8192"),
        ("qmc_ghk (runner contrasts) B=32768 (normalised)", "qmc_ghk B=32768")]
print("\n### max |p - truth| over all winners, normalised estimates\n")
print("| method | " + " | ".join(col(k) for k in fields) + " |")
print("|---|" + "---:|" * len(fields))
for m, label in want:
    cells = []
    for k in fields:
        r = by.get((k[0], k[1], k[2], m)); cells.append(f"{float(r['err']):.1e}" if r else "-")
    print(f"| {label} | " + " | ".join(cells) + " |")

print("\n### Wall time, seconds (Apple M3 Ultra, 2 BLAS threads, numpy lattice path, LPs via HiGHS)\n")
print("| method | " + " | ".join(col(k) for k in fields) + " |")
print("|---|" + "---:|" * len(fields))
for m, label in [("winning default rule", "default rule"), ("qmc_ghk (runner contrasts) B=32768 (normalised)", "qmc_ghk B=32768"),
                 ("Hybrid A per-winner R=2048 (normalised)", "Hybrid A per-winner R=2048"), ("Hybrid A shared R=16384 (normalised)", "Hybrid A shared R=16384")]:
    print(f"| {label} | " + " | ".join(f"{float(by[(k[0], k[1], k[2], m)]['seconds']):.2f}" for k in fields) + " |")

print("\n### Raw sums of the hybrid estimates (mass leaking past the margin shows here)\n")
print("| method | " + " | ".join(col(k) for k in fields) + " |")
print("|---|" + "---:|" * len(fields))
for m, label in [("Hybrid A per-winner R=2048 (raw)", "per-winner R=2048"), ("Hybrid A shared R=16384 (raw)", "shared R=16384")]:
    print(f"| {label} | " + " | ".join(f"{float(by[(k[0], k[1], k[2], m)]['raw_sum']):.4f}" for k in fields) + " |")
