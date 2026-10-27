"""Independent checks of all retained outputs, including metadata and minimum norm."""
import csv, json, pathlib, sys, tomllib
import numpy as np
from scipy import linalg as la
from common import CASES, load, read, factors, EPS

root=pathlib.Path(sys.argv[1]);phase=sys.argv[2]
reps=[0] if phase=='pilot' else range(1,21)
records=[];pairs=[];errors=[]
def rank(A):
    s=la.svdvals(A)
    return int((s>min(A.shape)*EPS*s[0]).sum()) if len(s) else 0
def relative(A,B):return float(la.norm(A-B)/max(1.,la.norm(B)))
for case in CASES:
    for rep in reps:
        label=f'{case}-{rep}';folder=root/'inputs'/label
        m,X,Y=load(folder);f=factors(X,Y);A,Z=f['A'],f['Z']
        variants={}
        for variant in ('baseline','candidate','python','r'):
            dest=root/'results'/phase/label/variant
            try:
                if variant in ('baseline','candidate'):
                    md=tomllib.loads((dest/'metadata.toml').read_text())
                    assert md['julia_threads']==md['blas_threads']==1
                elif variant=='python':
                    md=json.loads((dest/'metadata.json').read_text())
                    assert all(t['num_threads']==1 for t in md['threadpools'])
                else:
                    md=next(csv.DictReader((dest/'metadata.csv').open()))
                for key in ('X','Y'):
                    kind='md5' if variant=='r' else 'sha256'
                    assert md[f'{key}_{kind}']==m[f'{key}_{kind}']
                state={}
                modes=[('fit',3)]+[('path',r) for r in range(4)]
                if variant!='r':modes+=[('coefficient-fit',3)]+[(f'coefficient-path-{r}',r) for r in range(4)]
                for prefix,r in modes:
                    path=dest/(f'{prefix}-{r}-B.bin' if prefix in ('fit','path') else f'{prefix}.bin')
                    B=read(path,(m['p'],m['q']));state[(prefix,r)]=B
                    reference=read(folder/f'reference-{r}.bin',B.shape)
                    E=Z-A@B;obj=float(np.sum(E**2)/(2*m['n']))
                    gap=abs(obj-m['reference_objectives'][r])/max(1.,abs(m['reference_objectives'][r]))
                    rk=rank(B);minnorm=relative(B,f['V']@(f['V'].T@B))
                    coefficient_error=relative(B,reference)
                    prediction_error=relative(A@B,A@reference)
                    good=bool(np.all(np.isfinite(B)) and gap<=1e-8 and rk<=r and prediction_error<=1e-7)
                    # R's pivoted QR may choose a non-minimum-norm representative.
                    if variant!='r' or f['xrank']==m['p']:
                        good=good and minnorm<=1e-8 and coefficient_error<=1e-7
                    row=dict(case=case,rep=rep,variant=variant,mode=prefix,rank_bound=r,rank=rk,
                        objective=obj,relative_loss_error=gap,relative_coefficient_error=coefficient_error,
                        relative_prediction_error=prediction_error,relative_minimum_norm_error=minnorm,
                        residual_error=0.,intercept_error=0.,df_error=0.,exact_df_error=0.,passed=good)
                    if variant!='r' and prefix in ('fit','path'):
                        filename=dest/f'{prefix}-{r}'
                        data=tomllib.loads(filename.with_suffix('.toml').read_text()) if variant!='python' else json.loads(filename.with_suffix('.json').read_text())
                        returned=read(dest/f'{prefix}-{r}-E.bin',E.shape)
                        intercept=read(dest/f'{prefix}-{r}-intercept.bin',(m['q'],))
                        residual_error=relative(returned,E)
                        intercept_error=relative(intercept,Y.mean(0)-B.T@X.mean(0))
                        support=np.flatnonzero(np.any(B!=0,axis=1))
                        k=rank(A[:,support]) if len(support) else 0
                        df=rk*(k+m['q']-rk)+m['q']
                        exact=m['exact_df'][r]
                        df_error=abs(data['df']-df)
                        exact_error=abs(data['df_exact']-exact)/max(1.,abs(exact))
                        metadata_ok=(data['rank']==rk and data['support']==(support+1).tolist() and data['converged'])
                        objective_error=abs(data['objective']-obj)/max(1.,abs(obj))
                        good=good and metadata_ok and df_error==0 and exact_error<=1e-8 and residual_error<=1e-8 and intercept_error<=1e-8 and objective_error<=1e-8
                        row.update(residual_error=residual_error,intercept_error=intercept_error,df_error=df_error,exact_df_error=exact_error,passed=bool(good))
                    records.append(row)
                variants[variant]=state
            except Exception as exc:
                errors.append(dict(case=case,rep=rep,variant=variant,error=repr(exc)))
        if 'baseline' in variants and 'candidate' in variants:
            for key,B in variants['baseline'].items():
                C=variants['candidate'][key]
                pairs.append(dict(case=case,rep=rep,mode=key[0],rank=key[1],
                    exact_coefficients=bool(np.array_equal(B,C)),relative_coefficient_error=relative(B,C),
                    passed=relative(B,C)<=1e-12))
result=root/'results'/phase
for name,rows in [('certificates.csv',records),('paired-certificates.csv',pairs)]:
    with (result/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['case','rep','variant','mode','passed']);w.writeheader();w.writerows(rows)
expected_cases=len(CASES)*len(reps)
summary=dict(shared_cases=expected_cases,output_records=len(records),expected_output_records=35*expected_cases,
    failed_certificates=sum(not x['passed'] for x in records),paired_records=len(pairs),
    expected_paired_records=10*expected_cases,failed_pairs=sum(not x['passed'] for x in pairs),
    bitwise_equal_pairs=sum(x['exact_coefficients'] for x in pairs),errors=errors)
(result/'validation.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
assert len(records)==summary['expected_output_records'] and len(pairs)==summary['expected_paired_records'] and not errors
assert summary['failed_certificates']==summary['failed_pairs']==0
