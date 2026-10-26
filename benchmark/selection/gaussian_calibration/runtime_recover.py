"""Prepare stage recovery under Slurm, preserving every original timing row."""
import csv
import os
from pathlib import Path
import shutil
from study_check import fits

base=Path(os.environ['HDM_BASE'])
old=base/'runtime-pilot-v1';new=Path(os.environ['HDM_STUDY'])
assert not new.exists()
new.mkdir();(new/'inputs').symlink_to(old/'inputs',target_is_directory=True)
(new/'fits').mkdir();(new/'logs').mkdir()
for engine in ('native','julia','R'):
    for sc in range(1,5):
        name=f'{engine}-sc{sc}-shard0';source=old/'fits'/name;target=new/'fits'/name
        if Path(str(source)+'.done').exists():
            for p in (old/'fits').glob(name+'*'):(new/'fits'/p.name).symlink_to(p,target_is_directory=p.is_dir())
            continue
        assert engine=='native'
        metadata=list(csv.DictReader(Path(str(source)+'.records.csv').open()))
        binary=list(fits(Path(str(source)+'.bin')))
        assert len(metadata)==len(binary), 'incomplete binary/meta pair; manual reconciliation required'
        assert all(int(m['seed'])==b[0] and int(m['ncolumns'])==b[2].shape[1] for m,b in zip(metadata,binary))
        for suffix in ('.bin','.records.csv','.timings.csv','.witness.csv'):
            shutil.copy2(Path(str(source)+suffix),Path(str(target)+suffix))
        # A later timing record proves the previous allocation dump closed.
        # Last active workload has no completion proof; reprofile it only.
        last=metadata[-1]
        last_key=(last['seed'],last['method'],last['payload'],last['setting'])
        profiles=Path(str(target)+'-profiles');profiles.mkdir()
        for m in metadata:
            key=m['seed'],m['method'],m['payload'],m['setting']
            if key==last_key or m['setting'] not in ('public','prepared'):continue
            # Dataset index is fixed by the three declared runtime seeds.
            dataset=int(m['seed'])%100
            prefix=f"{dataset}-{m['method']}-{m['payload']}-{m['setting']}"
            cpu=Path(str(source)+'-profiles')/(prefix+'.cpu.txt')
            alloc=Path(str(source)+'-profiles')/(prefix+'.alloc.txt')
            if cpu.exists() and alloc.exists():
                for p in (cpu,alloc):
                    dest=profiles/p.name
                    if not dest.exists():dest.symlink_to(p)
                (profiles/(prefix+'.done')).write_text('prior workload completed before a later timing record\n')
        print(name,'retained_timing_records',len(metadata),flush=True)
