import subprocess, sys, re
S = sys.argv[1]; which = sys.argv[2]
shas = ["8e1815f4"] + open(f"{S}/fp.txt").read().split()
cache = {}
def bad(i):
    if i in cache: return cache[i]
    out = subprocess.run([f"{S}/fish_at.sh", shas[i]], capture_output=True, text=True).stdout.strip()
    print(i, out, flush=True)
    if which == "py":
        m = re.search(r"referee logLik (-[\d.]+)", out); b = (m is None) or float(m.group(1)) < -1213.5
    else:
        b = "julia exact logLik" not in out
    cache[i] = b; return b
lo, hi = 0, len(shas) - 1   # lo good, hi bad
assert not bad(lo) and bad(hi)
while hi - lo > 1:
    mid = (lo + hi) // 2
    if bad(mid): hi = mid
    else: lo = mid
print("FIRST BAD", shas[hi], "last good", shas[lo])
