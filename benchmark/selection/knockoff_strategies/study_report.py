"""Fixed-cohort reporting and acceptance checks; execute under Slurm only."""
import csv,json,os,sys
from pathlib import Path
import numpy as np
root=Path(os.environ['HDM_KNOCKOFF_STUDY']);shards=[int(v) for v in sys.argv[1].split(',')];count=int(sys.argv[2])
SCENARIOS=['iid-alt','iid-same','ar04-alt','ar04-same','ar08-alt','ar08-same','weak-ar04-alt','weak-ar04-same',
 'wide-ar04-alt','wide-ar04-same','null-iid','null-ar04','null-ar08','misspecified-iid','misspecified-ar04','misspecified-ar08']
methods=('equicorrelated','mvr','maxent');rows=[];statuses=[];rrows=[];rstatuses=[];input_records=[]
for sc in range(1,17):
    for shard in shards:
        out=root/f'scenario-{sc}-shard-{shard}'
        cert=list(csv.DictReader((out/'certificates.csv').open()));assert len(cert)==3*count
        for r in cert:
            for key in ('scenario','shard','dataset','seed','selected','true','false'):r[key]=int(r[key])
            for key in ('residual','objective','fdp','power','threshold'):r[key]=float(r[key])
            r['valid']=r['valid']=='True';r['scenario_name']=SCENARIOS[sc-1]
        rows+=cert
        status=list(csv.DictReader((out/'fit-status.csv').open()));assert len(status)==3*count;statuses+=status
        input_records.append(dict(scenario=sc,shard=shard,expected_inputs=count))
        prefix=os.environ.get('HDM_R_PREFIX','R')
        rfile=out/f'{prefix}-certificates.csv'
        if rfile.exists():
            rs=list(csv.DictReader(rfile.open()));assert len(rs)==3*min(3,count);rrows+=rs
            rstatuses+=list(csv.DictReader((out/f'{prefix}-status.csv').open()))

N=16*len(shards)*count;assert len(rows)==3*N
assert len({r['seed'] for r in rows})==N
# The three procedures share each dataset, and are never counted as independent
# replications of one another. Check every public-filter forwarding witness.
api=[r for r in statuses if not r['api_match']=='true'];flags=[r for r in statuses if r['converged']!='true'];errors=[r for r in statuses if r['success']!='true']

def summarize(values):
    x=np.asarray(values,float);x=x[np.isfinite(x)]
    if not len(x):return (float('nan'),)*4
    mean=float(np.mean(x));se=float(np.std(x,ddof=1)/np.sqrt(len(x))) if len(x)>1 else float('nan')
    return mean,se,mean-1.96*se,mean+1.96*se

def wilson(success,total):
    z=1.96;rate=success/total;den=1+z*z/total
    center=(rate+z*z/(2*total))/den;radius=z*np.sqrt(rate*(1-rate)/total+z*z/(4*total*total))/den
    return float(center-radius),float(center+radius)

mc=[];paired=[];worker=[]
for sc,name in enumerate(SCENARIOS,1):
    groups={m:{r['seed']:r for r in rows if r['scenario']==sc and r['strategy']==m} for m in methods}
    for m,group in groups.items():
        valid=[r for r in group.values() if r['valid']];total=len(group);nv=len(valid)
        record=dict(scenario=sc,name=name,strategy=m,inputs=total,valid=nv,failures=total-nv)
        for metric in ('fdp','power','selected','true','false'):
            mean,se,lo,hi=summarize([r[metric] for r in valid]);record.update({f'{metric}_mean':mean,f'{metric}_se':se,f'{metric}_lo95':lo,f'{metric}_hi95':hi})
        record['fdr_failure_lower_bound']=sum(r['fdp'] for r in valid)/total
        record['fdr_failure_upper_bound']=(sum(r['fdp'] for r in valid)+total-nv)/total
        if sc in (11,12,13) and nv:
            lo,hi=wilson(sum(r['selected']>0 for r in valid),nv);record['fdp_lo95']=lo;record['fdp_hi95']=hi
        record['failure_lo95'],record['failure_hi95']=wilson(total-nv,total)
        mc.append(record)
        for shard in shards:
            subset=[r for r in valid if r['shard']==shard]
            worker.append(dict(scenario=sc,name=name,shard=shard,strategy=m,valid=len(subset),
                               fdr=summarize([r['fdp'] for r in subset])[0],power=summarize([r['power'] for r in subset])[0]))
    for m,other in (('mvr','equicorrelated'),('maxent','equicorrelated'),('mvr','maxent')):
        common=[seed for seed in groups[m] if groups[m][seed]['valid'] and groups[other][seed]['valid']]
        for metric in ('fdp','power','selected','true','false'):
            mean,se,lo,hi=summarize([groups[m][seed][metric]-groups[other][seed][metric] for seed in common])
            paired.append(dict(scenario=sc,name=name,method=m,reference=other,metric=metric,paired_inputs=len(common),
                               difference=mean,se=se,lo95=lo,hi95=hi))

def write_csv(name,data):
    with (root/name).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)

write_csv('monte-carlo.csv',mc);write_csv('paired-differences.csv',paired);write_csv('worker-means.csv',worker)
failures=[r for r in rows if not r['valid']]
summary=dict(inputs=N,independent_inputs_per_scenario=len(shards)*count,scenarios=16,methods=3,certificates=len(rows),
             valid_certificates=len(rows)-len(failures),max_residual=max(r['residual'] for r in rows),failures=failures,
             public_api_witness_failures=api,native_stopping_flags=flags,native_fit_errors=errors,
             R_certificates=len(rrows),R_valid=sum(r['valid']=='True' for r in rrows),
             R_selection_disagreements=sum(int(r['selection_disagreement'])>0 for r in rrows),
             R_max_residual=max((float(r['residual']) for r in rrows),default=None),
             R_max_relative_objective_difference=max((float(r['relative_objective_difference']) for r in rrows),default=None),
             R_max_relative_coefficient_difference=max((float(r['relative_coefficient_difference']) for r in rrows),default=None),
             R_fit_errors=[r for r in rstatuses if r['success']!='TRUE'])
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# Shared-data knockoff construction evidence','',
 f'{N:,} independent inputs across 16 scenarios, {len(shards)*count:,} per scenario; three paired strategies per input. Pilot shard zero is excluded from main inference. Known population covariance is used in scenarios1–13; scenarios14–16 deliberately misspecify it. All procedures share X,y,innovations,lambda=.04 and knockoff+q=.1. The lasso uses cyclic order,tol1e-9,max100000; independent original-design accuracy threshold is1e-7. No per-dataset tuning.','',
 '| Scenario | Strategy | Valid / total | FDR | MC SE | Power | MC SE | Mean false | Mean true |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
for r in mc:
    lines.append(f"| {r['name']} | {r['strategy']} | {r['valid']}/{r['inputs']} | {r['fdp_mean']:.4f} | {r['fdp_se']:.4f} | {r['power_mean']:.4f} | {r['power_se']:.4f} | {r['false_mean']:.3f} | {r['true_mean']:.3f} |")
lines+=['','Global-null power is undefined. Null FDR intervals use Wilson binomial intervals for nonempty selection; other intervals use dataset-level Monte Carlo SEs. Failed inputs are retained, with conservative FDP failure bounds in monte-carlo.csv. Paired differences are computed only on paired valid inputs with their denominator explicit. Separate worker means and every certificate remain available. The estimands are specific to these laws; misspecification, same-sign/alternating differences, severe-correlation and wide failures must remain in conclusions.','',
 f'Native independent certificates: {summary["valid_certificates"]}/{summary["certificates"]}; maximum residual{summary["max_residual"]:.6g}. Native internal stopping flags:{len(flags)}; fit errors:{len(errors)}; failed public-filter witnesses:{len(api)}. Matched R fits on the first three inputs per shard/scenario:{summary["R_valid"]}/{summary["R_certificates"]} valid; selected-set disagreements:{summary["R_selection_disagreements"]}. R inputs are a subset of the primary cohort, not extra independent datasets. Native statistical fitting reuses the fixed population diagonal within a shard; construction cost is measured separately in the runtime study.','',
 '| Scenario | Contrast | Metric | Paired difference | 95% MC interval |','|---|---|---|---:|---|']
for r in paired:
    if r['metric'] not in ('fdp','power'):continue
    lines.append(f"| {r['name']} | {r['method']} - {r['reference']} | {r['metric']} | {r['difference']:.4f} | [{r['lo95']:.4f}, {r['hi95']:.4f}] |")
(root/'TABLES.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ('failures','native_stopping_flags','native_fit_errors','R_fit_errors')}))
assert not failures and not api and not errors,'native acceptance failures retained'
