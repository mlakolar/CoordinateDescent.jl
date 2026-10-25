"""Paired independent-input summaries. Run only under Slurm."""
import csv,json,pathlib,sys,tomllib
import numpy as np
from common import CASES
root=pathlib.Path(sys.argv[1]);phase=sys.argv[2];base=root/'results'/phase
reps=list(range(1,21)) if phase.startswith('main') else [0]
validation=json.loads((base/'validation.json').read_text());path_validation=json.loads((base/'path-validation.json').read_text())
rows=[];resources=[];profiles=[]
variants=['public-baseline','public-candidate','common-baseline','common-candidate','r-public','r-common','path-baseline','path-candidate']
for case in CASES:
 for rep in reps:
  for variant in variants:
   dest=base/f'{case}-{rep}'/variant
   if (dest/'timings.csv').exists():
    for r in csv.DictReader((dest/'timings.csv').open()):
     r['experiment']=variant;r['rep']=int(r['rep']);r['seconds']=float(r['seconds']);rows.append(r)
   if (dest/'resource.json').exists():
    resource=json.loads((dest/'resource.json').read_text())
    md=next(csv.DictReader((dest/'metadata.csv').open())) if variant.startswith('r-') and (dest/'metadata.csv').exists() else tomllib.loads((dest/'metadata.toml').read_text()) if (dest/'metadata.toml').exists() else {}
    resources.append(dict(case=case,rep=rep,experiment=variant,**resource,metadata=md))
   if case in ['wide','responses','ill']:
    for p in sorted(dest.glob('cpu-*.txt')):
     text=p.read_text();first=text.splitlines()[0] if text else ''
     profiles.append(dict(case=case,rep=rep,experiment=variant,file=str(p.relative_to(base)),header=first))
rng=np.random.default_rng(10226126)
def interval(values):
 a=np.asarray(values,float)
 if not len(a):return dict(inputs=0)
 boot=np.median(rng.choice(a,(10000,len(a)),replace=True),axis=1)
 return dict(inputs=len(a),ratio=float(np.median(a)),minimum=float(a.min()),maximum=float(a.max()),lo=float(np.quantile(boot,.025)),hi=float(np.quantile(boot,.975)))
def median_time(case,rep,experiment,tag):
 data=[r['seconds'] for r in rows if r['case']==case and r['rep']==rep and r['experiment']==experiment and r['config']==tag]
 if len(data)!=3:return None
 return float(np.median(data))
table=[]
for case in CASES:
 configs=[f'{s}-{f}-{w}' for s in ['cd','fista'] for f,w in ([('.5','unit'),('.1','unit'),('.02','unit')]+([('.1','adaptive')] if case=='medium' else []))]
 configs=[t.replace('-.','-0.') for t in configs]
 for mode in ['public','common']:
  for tag in configs:
   for comparator in [mode+'-baseline','r-'+mode]:
    pairs=[p for p in validation['pairs'] if p['case']==case and p['mode']==mode and p['config']==tag and p['comparator']==comparator]
    ratios=[];a=[];b=[];unmatched=[];missing=[]
    for rep in reps:
     pair=next((p for p in pairs if p['rep']==rep),None)
     otag='r-'+tag.split('-',1)[1] if comparator.startswith('r-') else tag
     c=median_time(case,rep,mode+'-candidate',tag);o=median_time(case,rep,comparator,otag)
     if pair is None or c is None or o is None:missing.append(rep);continue
     if not pair['matched_objective']:unmatched.append(dict(rep=rep,objective_relative_difference=pair['objective_relative_difference'],timing_ratio=c/o));continue
     ratios.append(c/o);a.append(c);b.append(o)
    table.append(dict(case=case,mode=mode,config=tag,comparator=comparator,candidate_ms=1000*float(np.median(a)) if a else None,comparator_ms=1000*float(np.median(b)) if b else None,
      **interval(ratios),unmatched=unmatched,missing=missing))
# Direct solver-default evidence: only compare attained-equivalent objectives.
solver_table=[]
cert={(r['case'],r['rep'],r['variant'],r['config']):r for r in validation['certificates']}
for case in CASES:
 for frac,weight in [('0.5','unit'),('0.1','unit'),('0.02','unit')]+([('0.1','adaptive')] if case=='medium' else []):
  ratios=[];unmatched=[]
  for rep in reps:
   ca,cb='fista-'+frac+'-'+weight,'cd-'+frac+'-'+weight
   ra,rb=cert.get((case,rep,'public-candidate',ca)),cert.get((case,rep,'public-candidate',cb))
   a=median_time(case,rep,'public-candidate',ca);b=median_time(case,rep,'public-candidate',cb)
   if ra is None or rb is None or a is None or b is None:continue
   gap=abs(ra['objective']-rb['objective'])/max(1,abs(ra['objective']),abs(rb['objective']))
   if gap>1e-6:unmatched.append(dict(rep=rep,gap=gap));continue
   ratios.append(a/b)
  solver_table.append(dict(case=case,penalty=frac,weights=weight,**interval(ratios),unmatched=unmatched))
path_table=[]
for case in CASES:
 for tag in [s+'-'+w for s in ['cd','fista'] for w in (['unit','adaptive'] if case=='medium' else ['unit'])]:
  ratios=[];a=[];b=[];unmatched=[]
  for rep in reps:
   ps=[p for p in path_validation['pairs'] if p['case']==case and p['rep']==rep and p['config'].rsplit('-',1)[0]==tag]
   c=median_time(case,rep,'path-candidate',tag);o=median_time(case,rep,'path-baseline',tag)
   if len(ps)!=8 or c is None or o is None:continue
   if not all(p['matched_objective'] for p in ps):unmatched.append(rep);continue
   ratios.append(c/o);a.append(c);b.append(o)
  path_table.append(dict(case=case,config=tag,candidate_ms=1000*float(np.median(a)) if a else None,baseline_ms=1000*float(np.median(b)) if b else None,**interval(ratios),unmatched=unmatched))
component_rows=list(csv.DictReader((base/'components.csv').open()));component_table=[]
for case,solver,component in sorted(set((r['case'],r['solver'],r['component']) for r in component_rows)):
 ratios=[];a=[];b=[]
 for rep in reps:
  values={v:[float(r['seconds']) for r in component_rows if r['case']==case and int(r['rep'])==rep and r['solver']==solver and r['component']==component and r['variant']==v] for v in ['baseline','candidate']}
  if all(len(z)==3 for z in values.values()):
   c,o=float(np.median(values['candidate'])),float(np.median(values['baseline']));ratios.append(c/o);a.append(c);b.append(o)
 component_table.append(dict(case=case,solver=solver,component=component,candidate_ms=1000*float(np.median(a)),baseline_ms=1000*float(np.median(b)),**interval(ratios)))
def writecsv(name,items):
 if items:
  with (base/name).open('w') as f:
   fields=list(dict.fromkeys(k for row in items for k in row));w=csv.DictWriter(f,fields,lineterminator='\n');w.writeheader();w.writerows(items)
writecsv('all-timings.csv',rows);writecsv('comparison.csv',table);writecsv('paths.csv',path_table);writecsv('solver-comparison.csv',solver_table);writecsv('component-summary.csv',component_table)
summary=dict(independent_inputs_per_design=len(reps),timing_rows=len(rows),expected_timing_rows=654*len(reps),processes=len(resources),expected_processes=48*len(reps),
 table=table,paths=path_table,solver_comparison=solver_table,components=component_table,resources=resources,profiles=profiles,
 validation={k:v for k,v in validation.items() if k not in ['certificates','pairs','metadata']},path_validation={k:v for k,v in path_validation.items() if k not in ['certificates','pairs']})
(base/'summary.json').write_text(json.dumps(summary,indent=2))
lines=['# Sparse reduced-rank setup campaign','',f'{len(reps)} independent shared input(s) per design; three warmed rounds per method/input. Paired bootstrap intervals resample independent inputs (10,000 draws, frozen seed).',
 'Stationarity is local. Objective-equivalence tolerance is 1e-6 relative; unmatched outcomes are listed separately in summary.json. The R adapter uses historical coefficient-change tolerances (1e-8 outer, 1e-9 inner), because its rules differ from Julia KKT stopping.',
 'Public fits include natural initialization and required diagnostics. Shared-initialization experiments exclude centering/RRR/SVD initialization; Julia includes sufficient statistics and dense result diagnostics, R includes its native design-rank/IC diagnostics. Component probes use cached inputs and are not additive public-fit decompositions.','',
 '| design | mode | config | comparator | candidate ms | comparator ms | ratio [95% interval] | matched inputs | unmatched |',
 '|---|---|---|---|---:|---:|---|---:|---:|']
for r in table:
 if r['inputs']:lines.append(f"|{r['case']}|{r['mode']}|{r['config']}|{r['comparator']}|{r['candidate_ms']:.3f}|{r['comparator_ms']:.3f}|{r['ratio']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]|{r['inputs']}|{len(r['unmatched'])}|")
lines+=['','Full path, solver/default, components, memory/process metadata, profiles, failed certificates and unmatched outcomes are in the adjacent tables and JSON. Whole-process RSS includes runtime/imports/JIT, not just optimizer workspace.']
(base/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:summary[k] for k in ['independent_inputs_per_design','timing_rows','expected_timing_rows','processes','expected_processes','validation','path_validation']},indent=2))
assert len(rows)==summary['expected_timing_rows'] and len(resources)==summary['expected_processes']
