import cProfile, csv, hashlib, json, pathlib, pstats, resource, sys, time
import numpy as np, scipy
from scipy import linalg as la
from threadpoolctl import threadpool_info, threadpool_limits
from common import load, factors, coefficient, save, EPS

folder,out=map(pathlib.Path,sys.argv[1:3]);out.mkdir(parents=True,exist_ok=True)
m,X,Y=load(folder)
def exact(f,r):
    q=f['Z'].shape[1];base=r*(f['xrank']+q-r)+q
    if r==0:return float(base)
    ratio=f['h'][None,r:]/f['h'][:r,None];gap=1-ratio**2
    return float('inf') if np.any(gap<=EPS) else float(base+np.sum(2*ratio**2/gap))
def julia_rank(A):
    s=la.svdvals(A)
    return int((s>min(A.shape)*EPS*s[0]).sum()) if s.size else 0
def solution(f,r):
    p,q=f['A'].shape[1],f['Z'].shape[1]
    if r==0:B=np.zeros((p,q));rk=0;E=f['Z'].copy()
    else:
        L=f['V']@(f['H'][:,:r]/f['s'][:,None]);R=f['h'][:r,None]*f['Wt'][:r]
        B=L@R
        qL=la.qr(L,mode='economic')[1];qR=la.qr(R.T,mode='economic')[1]
        s=la.svdvals(qL@qR.T);cutoff=min(p,q)*EPS*s[0]
        ambiguous=np.any(s<=8*cutoff) or 8*r*EPS*la.norm(L)*la.norm(R)>cutoff
        rk=julia_rank(B) if ambiguous else int((s>cutoff).sum())
        E=f['Z']-(f['A']@L)@R if r*(p+q)<p*q else f['Z']-f['A']@B
    support=np.flatnonzero(np.any(B!=0,axis=1))
    k=0 if len(support)==0 else f['diagnostic_xrank'] if len(support)==p else julia_rank(f['A'][:,support])
    return dict(B=B,E=E,intercept=f['my']-B.T@f['mx'],rank=rk,support=(support+1).tolist(),
                df=int(rk*(k+q-rk)+q),df_exact=exact(f,r),objective=float(np.sum(E**2)/(2*len(f['A']))),converged=True)
def public_fit():return solution(factors(X,Y),3)
def public_path():
    f=factors(X,Y);return [solution(f,r) for r in range(4)]
def coefficient_fit():return coefficient(factors(X,Y),3)
def coefficient_path():
    f=factors(X,Y);return [coefficient(f,r) for r in range(4)]
modes=dict(public_fit=public_fit,public_path=public_path,coefficient_fit=coefficient_fit,coefficient_path=coefficient_path)
def dump(s,r,prefix):
    for name in ('B','E','intercept'):save(out/f'{prefix}-{r}-{name}.bin',s[name])
    (out/f'{prefix}-{r}.json').write_text(json.dumps({k:v for k,v in s.items() if k not in ('B','E','intercept')},indent=2))
def persist(value,mode):
    if mode=='public_fit':dump(value,3,'fit')
    elif mode=='public_path':
        for r,s in enumerate(value):dump(s,r,'path')
    elif mode=='coefficient_fit':save(out/'coefficient-fit.bin',value)
    else:
        for r,B in enumerate(value):save(out/f'coefficient-path-{r}.bin',B)
def rss():
    return int(next(x.split()[1] for x in pathlib.Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:')))*1024
with threadpool_limits(limits=1):
    for fn in modes.values():fn()
    meta=dict(variant='python',case=m['case'],rep=m['rep'],python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
              threadpools=threadpool_info(),warmed_current_rss=rss(),warmed_peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    for name in ('X','Y'):meta[f'{name}_sha256']=hashlib.sha256((folder/f'{name}.bin').read_bytes()).hexdigest()
    meta['source_sha256']=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
    with (out/'timings.csv').open('w') as f:
        w=csv.writer(f);w.writerow(('variant','case','rep','mode','round','seconds','bytes','gctime'))
        names=list(modes)
        for rd in range(1,4):
            offset=(m['rep']+rd)%len(names)
            for mode in names[offset:]+names[:offset]:
                start=time.perf_counter();value=modes[mode]();elapsed=time.perf_counter()-start
                w.writerow(('python',m['case'],m['rep'],mode,rd,elapsed,'',''));f.flush()
                if rd==1:persist(value,mode)
    meta['peak_after_timing']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    meta['current_after_timing']=rss()
    if m['case'] in ('responses','wide'):
        for mode in ('public_fit','public_path','coefficient_fit'):
            profiler=cProfile.Profile();began=time.perf_counter();calls=0;profiler.enable()
            while calls==0 or (calls<20 and time.perf_counter()-began<.25):modes[mode]();calls+=1
            profiler.disable()
            with (out/f'cpu-{mode}.txt').open('w') as f:
                f.write(f'calls={calls}\n');pstats.Stats(profiler,stream=f).strip_dirs().sort_stats('cumtime').print_stats(80)
    meta['whole_worker_peak_rss']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    (out/'metadata.json').write_text(json.dumps(meta,indent=2))
