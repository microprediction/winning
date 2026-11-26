"""Run in research/shared_field_research, OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u bench_sparse_cavity.py"""
import numpy as np
from time import perf_counter
from sparse_cavity import prepare,shares
from bench_markov_tree import ghk
from bench_packed_split import case


def run():
    for n,kind in [(32,'diffuse'),(64,'diffuse'),(32,'wishart'),(32,'hetero'),(64,'clone'),(128,'diffuse'),(128,'weak')]:
        mu,C=case(n,kind); pp={}
        for pair in [False,True]:
            t=perf_counter(); prep=prepare(C,pair); pp[pair]=prep
            print(n,kind,pair,'prep',round(perf_counter()-t,5),'median sd',np.median(prep[-1]),flush=True)
        refp=np.array([ghk(mu,C,13 if n==32 else 12,s+100) for s in range(3)]);ref=refp.mean(0)
        print('ref spread',np.ptp(refp,axis=0).max(),'mass',ref.sum(),flush=True)
        for label,m in [('ghk',10),('single',14),('pair',14),('pair',16),('direct',16)]:
            t=perf_counter(); est=np.array([ghk(mu,C,m,s) if label=='ghk' else shares(mu,pp[label=='pair'],m,s,mode='direct' if label=='direct' else 'cavity') for s in range(4)]); sec=(perf_counter()-t)/4
            print(label,m,round(sec,5),'medmax',np.median(abs(est-ref).max(1)),'rmse',np.sqrt(np.mean((est-ref)**2)),'mass',np.max(abs(est.sum(1)-1)),flush=True)
if __name__=='__main__':run()
