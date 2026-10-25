import csv,json,pathlib,re,statistics,sys
root=pathlib.Path(sys.argv[1]);base=root/'results'/'main';summary=json.loads((base/'summary.json').read_text());val=json.loads((base/'validation.json').read_text());pv=json.loads((base/'path-validation.json').read_text())
expected={'baseline':'b611a82eb5228e1f33ca2d1dce66aa682187764025519e648a759fa2461de448','candidate':'99a97cc5151408181ac880cc6bc3cd271b24d7b779deb36087ff77cf93bcd5fc'}
for r in summary['resources']:
 md=r['metadata'];variant=r['experiment'];assert '7513' in md['cpu'];assert r['exit_code']==0
 if variant.startswith('r-'):
  assert md['rrpack']=='0.1.14' and all(md[k]=='1' for k in ['openblas_threads','omp_threads','mkl_threads'])
 else:
  assert md['source_sha256']==expected[variant.rsplit('-',1)[1]] and md['blas_threads']==md['julia_threads']==1
memory=[]
for case in ['small','medium','responses','wide','ill','deficient']:
 for variant in ['public-baseline','public-candidate','common-baseline','common-candidate','r-public','r-common','path-baseline','path-candidate']:
  items=[r for r in summary['resources'] if r['case']==case and r['experiment']==variant];assert len(items)==20
  values={k:[float(r['metadata'][k]) for r in items] for k in ['warmed_current_rss','warmed_peak_rss','current_after_timing','peak_after_timing','whole_worker_peak_before_replay']}
  for key,a in values.items():memory.append(dict(case=case,experiment=variant,metric=key,median_bytes=statistics.median(a),minimum_bytes=min(a),maximum_bytes=max(a)))
counts=[]
for case in ['small','medium','responses','wide','ill','deficient']:
 for variant in ['public-baseline','public-candidate','common-baseline','common-candidate']:
  cs=[r for r in val['certificates'] if r['case']==case and r['variant']==variant]
  for config in sorted({r['config'] for r in cs}):
   rs=[r for r in cs if r['config']==config];assert len(rs)==20
   for metric in ['iterations','cd_sweeps','fista_steps','kkt_calls']:
    a=[r['probe'][metric] for r in rs];counts.append(dict(case=case,experiment=variant,config=config,metric=metric,median=statistics.median(a),minimum=min(a),maximum=max(a)))
profiles=[]
for record in summary['profiles']:
 path=base/record['file'];text=path.read_text();item=dict(record)
 hit=re.search(r'raw_stack_words=(\d+)',text);item['raw_stack_words']=int(hit[1]) if hit else None
 if record['experiment'].startswith('r-'):
  raw=path.with_suffix('.Rprof');lines=raw.read_text().splitlines();item['raw_profile_lines']=len(lines);item['sampling_header']=lines[0] if lines else ''
 profiles.append(item)
assert len(profiles)==2040
beforeafter=[p for p in val['pairs'] if p['comparator'].endswith('-baseline')]
wide=[p for p in val['pairs'] if p['case']=='wide' and p['config']=='cd-0.5-unit' and p['comparator']=='r-public']
report=dict(verified_process_metadata=len(summary['resources']),expected_source_sha256=expected,profiles=len(profiles),native_cpu_profiles=sum(not p['experiment'].startswith('r-') for p in profiles),R_cpu_memory_profiles=sum(p['experiment'].startswith('r-') for p in profiles),native_allocation_reports=240,
 zero_stack_reports=[p['file'] for p in profiles if p['raw_stack_words']==0],beforeafter_pairs=len(beforeafter),beforeafter_unmatched=sum(not p['matched_objective'] for p in beforeafter),
 max_beforeafter_coefficient_difference=max(p['coefficient_relative_difference'] for p in beforeafter),
 max_kkt=max(r['kkt'] for r in val['certificates']+pv['certificates']),max_response=max(r['response'] for r in val['certificates']+pv['certificates']),
 strong_wide_R_unmatched=[p for p in wide if not p['matched_objective']],memory=memory,iteration_counts=counts,profile_index=profiles,
 regressions=[r for r in summary['table'] if r['mode']=='public' and r['comparator']=='public-baseline' and r['inputs'] and r['lo']>1],
 default_interpretation='Keep block_solver=:cd and all statistical/stopping defaults; wide strong/middle FISTA slower than CD, ill/deficient middle/weak FISTA substantially faster. Document explicit FISTA guidance; no universal solver or condition-trigger default adopted from this numerical grid.')
(base/'poststudy.json').write_text(json.dumps(report,indent=2))
for name,items in [('memory-summary.csv',memory),('iteration-counts.csv',counts)]:
 with (base/name).open('w') as f:
  w=csv.DictWriter(f,list(items[0]),lineterminator='\n');w.writeheader();w.writerows(items)
print(json.dumps({k:v for k,v in report.items() if k not in ['memory','iteration_counts','profile_index','strong_wide_R_unmatched','regressions']},indent=2))
