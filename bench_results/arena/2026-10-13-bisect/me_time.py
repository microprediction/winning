import time, sys
from winning.methods import METHODS
from winning.bench.runner import problem
mu,V,D=problem('n200k2',200,2,1.0,102)
r={}
for name,b in (("mendell_elston",None),("ghk",1000),("smc_orthant",1000)):
    ts=[]
    for _ in range(3):
        t=time.perf_counter(); METHODS[name](mu,V,D,budget=b,seed=17); ts.append(time.perf_counter()-t)
    r[name]=round(min(ts),3)
print(r)
