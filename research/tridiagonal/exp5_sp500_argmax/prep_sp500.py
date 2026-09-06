"""The day of the market's yearly high, empirical side: from daily
S&P 500 closes (Yahoo chart API, ^GSPC, 1927-), the within-year
trading-day index of each year's maximum close, plus pooled drift and
vol of daily log returns. Emits a small JSON for the julia engine run.

  curl -A "Mozilla/5.0" "https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC?period1=-1325635200&period2=1790000000&interval=1d" -o gspc.json
  python prep_sp500.py gspc.json out.json
"""
import datetime
import json
import math
import sys

d = json.load(open(sys.argv[1]))
r = d["chart"]["result"][0]
ts = r["timestamp"]
close = r["indicators"]["quote"][0]["close"]
rows = [(datetime.datetime.fromtimestamp(t, datetime.UTC).year, c)
        for t, c in zip(ts, close) if c is not None]

years = {}
for y, c in rows:
    years.setdefault(y, []).append(c)

per_year = []
for y in sorted(years):
    cs = years[y]
    if len(cs) < 200:                       # partial first/last year
        continue
    idx = max(range(len(cs)), key=lambda i: cs[i])
    per_year.append({"year": y, "n": len(cs),
                     "frac": (idx + 0.5) / len(cs)})

rets = []
prev = None
for _, c in rows:
    if prev is not None and prev > 0 and c > 0:
        rets.append(math.log(c / prev))
    prev = c
mu = sum(rets) / len(rets)
sd = math.sqrt(sum((x - mu) ** 2 for x in rets) / (len(rets) - 1))

json.dump({"per_year": per_year, "mu": mu, "sd": sd,
           "n_years": len(per_year)}, open(sys.argv[2], "w"))
print(f"{len(per_year)} full years, pooled mu {mu:.6f} sd {sd:.6f} "
      f"(daily log returns, n={len(rets)})")
