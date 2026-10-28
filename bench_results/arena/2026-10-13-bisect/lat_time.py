import numpy as np, time
from winning.methods import METHODS
from winning.bench.runner import problem
out=[]
for pid,n,k,sp,seed in [('n50k2',50,2,1.0,101),('n200k2',200,2,1.0,102),('n1000k2',1000,2,1.5,103),('n200k3',200,3,1.0,104)]:
    mu,V,D=problem(pid,n,k,sp,seed); ts=[]
    for _ in range(7):
        t=time.perf_counter(); p,info=METHODS['lattice'](mu,V,D); ts.append(time.perf_counter()-t)
    out.append(f"{pid} first {ts[0]:.4f} best {min(ts):.4f}")
print(" | ".join(out))
