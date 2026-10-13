import subprocess, sys, re, ast
S=sys.argv[1]; P="/Users/petercotton/github/winning-bench-pre707"
shas=subprocess.run(["git","-C",P,"rev-list","--first-parent","--reverse","3b9eaa18..a7f7d9fd"],capture_output=True,text=True).stdout.split()
shas=["3b9eaa18"]+shas
cache={}
def t(i):
    if i not in cache:
        subprocess.run(["git","-C",P,"checkout","-q","--detach",shas[i]],check=True)
        env={"PYTHONPATH":P,"OMP_NUM_THREADS":"4","OPENBLAS_NUM_THREADS":"4","VECLIB_MAXIMUM_THREADS":"4","RAYON_NUM_THREADS":"4","PATH":"/usr/bin:/bin"}
        o=subprocess.run(["nice","-n","10","/Users/petercotton/github/winning-bench-oct13/.venv/bin/python",f"{S}/me_time.py"],capture_output=True,text=True,env=env,cwd=S)
        line=(o.stdout.strip().splitlines() or ["ERR "+o.stderr[-300:]])[-1]
        print(i,shas[i][:8],line,flush=True); cache[i]=line
    return cache[i]
def bad(i,key,thr):
    l=t(i)
    try: return ast.literal_eval(l)[key]>thr
    except Exception: return None
for key,thr in (("mendell_elston",1.4),("ghk",2.15),("smc_orthant",2.6)):
    lo,hi=0,len(shas)-1
    if bad(lo,key,thr) or not bad(hi,key,thr): print(key,"endpoints?"); continue
    while hi-lo>1:
        m=(lo+hi)//2; b=bad(m,key,thr)
        if b is None:  # unrunnable: step forward
            m2=m+1
            while m2<hi and bad(m2,key,thr) is None: m2+=1
            b=bad(m2,key,thr); m=m2 if m2<hi else m
            if b is None: break
        if b: hi=m
        else: lo=m
    print("FIRST BAD",key,hi,shas[hi],flush=True)
