import pathlib,sys,json,time,csv,hashlib,resource,cProfile,pstats,numpy as np,tomllib
from threadpoolctl import threadpool_info,threadpool_limits
from python_solver import setup,solve
threadpool_limits(1)
folder,method,out=pathlib.Path(sys.argv[1]),sys.argv[2],pathlib.Path(sys.argv[3]);out.mkdir(parents=True,exist_ok=True);m=json.loads((folder/'meta.json').read_text());n,p,q=(m[k] for k in ['n','p','q'])
X=np.fromfile(folder/'X.bin',dtype='<f8').reshape((n,p),order='F').copy();Y=np.fromfile(folder/'Y.bin',dtype='<f8').reshape((n,q),order='F').copy();cached=setup(X,Y);lm=m['lambda_max'];pathfracs=np.geomspace(.5,.02,8);records=[];outcomes=[];failures=[]
def rss():
 return int(next(l.split()[1] for l in pathlib.Path('/proc/self/status').read_text().splitlines() if l.startswith('VmRSS:')))*1024
def cold(mode,frac):return [solve(method,setup(X,Y) if mode=='public' else cached,frac*lm,np.zeros((p,q)))]
def path(mode):
 c=setup(X,Y) if mode=='public' else cached;state=np.zeros((p,q));fits=[]
 for frac in pathfracs:
  z=solve(method,c,frac*lm,state);fits.append(z);state=z['B'].copy()
  if not z['converged']:break
 return fits
def fingerprint(fits):return hashlib.sha256(b''.join(np.asarray(z['B'],dtype='<f8').ravel(order='F').tobytes() for z in fits)).hexdigest()
md=dict(worker_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),method=method,input_hashes={k:hashlib.sha256((folder/(k+'.bin')).read_bytes()).hexdigest() for k in ['X','Y']},python=sys.version,cpu=next(l.split(':',1)[1].strip() for l in pathlib.Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),threadpools=threadpool_info(),solver_source_sha256=hashlib.sha256(pathlib.Path(__file__).with_name('python_solver.py').read_bytes()).hexdigest())
workloads=[]
for mode in ['public','cached']:
 for frac in [.5,.1,.02]:workloads.append((mode,'cold',frac,lambda mode=mode,frac=frac:cold(mode,frac)))
 workloads.append((mode,'path',0.,lambda mode=mode:path(mode)))
def store(fits,mode,kind,frac,stream):
 for i,z in enumerate(fits):
  offset=stream.tell();np.asarray(z['B'],dtype='<f8').ravel(order='F').tofile(stream);np.asarray(z['intercept'],dtype='<f8').tofile(stream)
  records.append(dict(mode=mode,kind=kind,fraction=float(pathfracs[i] if kind=='path' else frac),lambda_value=float((pathfracs[i] if kind=='path' else frac)*lm),offset_bytes=offset,iterations=z['iterations'],converged=z['converged'],has_full_result=True,objective=z['objective'],rank=z['rank'],df=z['df'],rows=z['rows'],native_residual_error=0.))
with (out/'timings.csv').open('w') as timing,(out/'fits.bin').open('wb') as binary:
 writer=csv.writer(timing,lineterminator='\n');writer.writerow(['mode','kind','fraction','round','seconds','bytes','iterations','converged','coefficient_sha256'])
 for mode,kind,frac,call in workloads:
  key=f'{mode}/{kind}/{frac}'
  try:
   warm=call();md.setdefault('warmed_current_rss',rss());md.setdefault('warmed_peak_rss',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
   if not all(z['converged'] for z in warm) or (kind=='path' and len(warm)!=8):
    store(warm,mode,kind,frac,binary);failures.append(dict(workload=key,error='warmup nonconverged; repeats/profiles skipped'));continue
   hashes=[]
   for rd in range(1,4):
    began=time.perf_counter();fits=call();seconds=time.perf_counter()-began;h=fingerprint(fits);hashes.append(h);writer.writerow([mode,kind,frac,rd,seconds,'',sum(z['iterations'] for z in fits),all(z['converged'] for z in fits),h]);timing.flush()
   assert len(set(hashes))==1 and all(z['converged'] for z in fits)
   store(fits,mode,kind,frac,binary);outcomes.append(dict(workload=key,status='complete',hash=hashes[0]))
  except Exception as e:failures.append(dict(workload=key,error=repr(e)))
md['timing_current_rss']=rss();md['timing_peak_rss']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
for label,call in [('middle',lambda:cold('public',.1)),('path',lambda:path('public'))]:
 key='public/cold/0.1' if label=='middle' else 'public/path/0.0'
 if not any(r['workload']==key for r in outcomes):continue
 prof=cProfile.Profile();prof.runcall(call)
 with (out/(label+'-cpu.txt')).open('w') as f:pstats.Stats(prof,stream=f).strip_dirs().sort_stats('cumulative').print_stats(100)
md['whole_worker_peak_rss']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;md['outcomes']=outcomes;md['failures']=failures
for r in records:r['lambda']=r.pop('lambda_value')
(out/'fits.json').write_text(json.dumps(dict(records=records),indent=2));(out/'metadata.json').write_text(json.dumps(md,indent=2))
