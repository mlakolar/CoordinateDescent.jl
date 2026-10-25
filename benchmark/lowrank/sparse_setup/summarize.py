import csv,json,pathlib,statistics,sys,tomllib
root=pathlib.Path(sys.argv[1]);phase=sys.argv[2] if len(sys.argv)>2 else 'pilot';base=root/'results'/phase
rows=[];table=[]
for p in sorted(base.glob('*-*/*/timings.csv')):rows.extend(csv.DictReader(p.open()))
validation=json.loads((base/'validation.json').read_text())
for case,tag in sorted(set((r['case'],r['config']) for r in rows)):
    values={v:[r for r in rows if r['case']==case and r['config']==tag and r['variant']==v] for v in ('baseline','candidate')}
    if all(len(z)==3 for z in values.values()):
        b,c=[statistics.median(float(r['seconds']) for r in values[v]) for v in ('baseline','candidate')]
        pair=next((r for r in validation['pairs'] if r['case']==case and r['config']==tag),{})
        table.append(dict(case=case,config=tag,baseline_ms=1000*b,candidate_ms=1000*c,ratio=c/b,
            baseline_allocated=statistics.median(int(r['bytes']) for r in values['baseline']),candidate_allocated=statistics.median(int(r['bytes']) for r in values['candidate']),
            matched_objective=pair.get('matched_objective'),objective_difference=pair.get('objective_relative_difference'),coefficient_difference=pair.get('coefficient_relative_difference')))
with (base/'summary.csv').open('w') as f:
    w=csv.DictWriter(f,list(table[0]),lineterminator="\n");w.writeheader();w.writerows(table)
summary=dict(phase=phase,independent_inputs_per_case=1,timing_rows=len(rows),expected_timing_rows=228,
    table=table,validation={k:v for k,v in validation.items() if k not in ('certificates','pairs')})
(base/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2));assert len(rows)==228 and not validation['errors'] and validation['failed_certificates']==0
