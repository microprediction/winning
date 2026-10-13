import subprocess, sys, re
S = sys.argv[1]; lo, hi = int(sys.argv[2]), int(sys.argv[3])
shas = ["8e1815f4"] + open(f"{S}/fp.txt").read().split()
cache = {}
def run(i):
    if i not in cache:
        out = subprocess.run([f"{S}/fish_at.sh", shas[i]], capture_output=True, text=True).stdout.strip()
        print(i, out, flush=True); cache[i] = out
    return cache[i]
preds = {
 "ll": lambda o: (m := re.search(r"referee logLik (-[\d.]+)", o)) is None or float(m.group(1)) < -1215.5,
 "time": lambda o: (m := re.search(r" in ([\d.]+)s", o)) is None or float(m.group(1)) > 15,
 "jl": lambda o: "julia exact logLik" not in o,
}
for name, pred in preds.items():
    a, b = lo, hi
    if pred(run(a)) or not pred(run(b)): print(name, "endpoints not good/bad"); continue
    while b - a > 1:
        mid = (a + b) // 2
        if pred(run(mid)): b = mid
        else: a = mid
    print("FIRST BAD", name, b, shas[b], flush=True)
