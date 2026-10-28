import json, sys
old = [json.loads(l) for l in open(sys.argv[1])]
new = [json.loads(l) for l in open(sys.argv[2])]
key = lambda r: (r["problem"], r["method"], r["budget"])
O = {key(r): r for r in old}
print("| problem | method | budget | sec old | sec new | ratio | max_err old | max_err new | log-odds err old | new | flag |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for r in new:
    o = O.get(key(r))
    if o is None:
        print(f"| {r['problem']} | {r['method']} | {r['budget']} | - | {r['seconds']:.3f} | new | - | {r['max_err']:.2e} | - | {r['max_log_err']:.3f} | NEW |"); continue
    ratio = r["seconds"] / max(o["seconds"], 1e-4)
    flags = []
    if ratio > 2 and r["seconds"] > 0.05: flags.append("TIME>2x")
    if r["max_err"] > 1.2 * o["max_err"] and r["max_err"] - o["max_err"] > 1e-4: flags.append("ABS-ERR")
    if r["max_log_err"] > 1.2 * o["max_log_err"] and r["max_log_err"] - o["max_log_err"] > 5e-3: flags.append("LOG-ERR")
    if r["max_err"] < 0.8 * o["max_err"] and o["max_err"] - r["max_err"] > 1e-4: flags.append("better")
    print(f"| {r['problem']} | {r['method']} | {r['budget']} | {o['seconds']:.3f} | {r['seconds']:.3f} | {ratio:.2f} | {o['max_err']:.2e} | {r['max_err']:.2e} | {o['max_log_err']:.3f} | {r['max_log_err']:.3f} | {' '.join(flags)} |")
for k in O:
    if k not in {key(r) for r in new}: print("| %s | %s | %s | MISSING IN NEW |" % k)
