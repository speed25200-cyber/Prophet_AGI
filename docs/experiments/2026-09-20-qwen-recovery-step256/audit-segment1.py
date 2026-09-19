import hashlib,json,math,sys
from pathlib import Path
import torch
torch.set_num_threads(2)
run=Path(sys.argv[1]); out=Path(sys.argv[2]); baseline=Path(sys.argv[3])
assert not out.exists()
report_path=run/'evaluation-step-000256.json'
report=json.loads(report_path.read_bytes())
manifest=json.loads((run/'recovery.json').read_bytes())
meta=report['checkpoint']
assert meta['slot'] in (0,1) and meta['step']==256
assert meta in json.loads((run/'manifest.json').read_bytes())['checkpoints']
checkpoint=run/('ckpt_slot'+str(meta['slot'])+'.pt')
with checkpoint.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
assert digest==meta['sha256'] and checkpoint.stat().st_size==meta['bytes']
state=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
canonical=lambda value:json.loads(json.dumps(value))
assert canonical(state['training_contract'])==manifest['training_contract']
assert canonical(state['config'])==manifest['config']
assert state['training_contract']['run_identity']==report['identity']
assert state['step']==report['step']==256
assert state['tokens_seen']==report['tokens_seen']==524288
assert state['skipped_nonfinite']==report['skipped_nonfinite']==state['consecutive_nonfinite']==0
assert state['training_contract']['dtype']=='float32' and not state['training_contract']['allow_tf32']
assert len(state['cuda_rng'])==1 and state['cuda_rng'][0].numel()>0
counts={'model':0,'optimizers':0}
def inspect(tree,group):
    if isinstance(tree,torch.Tensor):
        counts[group]+=1
        if tree.is_floating_point() or tree.is_complex():
            flat=tree.reshape(-1)
            assert all(bool(torch.isfinite(flat[i:i+1048576]).all()) for i in range(0,flat.numel(),1048576))
    elif isinstance(tree,dict):
        for value in tree.values(): inspect(value,group)
    elif isinstance(tree,(list,tuple)):
        for value in tree: inspect(value,group)
    elif isinstance(tree,float): assert math.isfinite(tree)
for group in counts: inspect(state[group],group)
rows=[json.loads(line) for line in (run/'training.jsonl').read_text().splitlines()]
assert len(rows)==256
for step,row in enumerate(rows,1):
    assert row['step']==step and row['tokens']==step*2048
    assert math.isfinite(row['loss']) and math.isfinite(row['extra']['train/grad_norm'])
    assert row['extra'].get('train/skipped_nonfinite',0)==0
before=json.loads(baseline.read_bytes())['evaluation']; after=report['evaluation']
for key in ('seq_len','batch_size','precision','loop_k','protocol','scored_tokens','scored_bytes'):
    assert before[key]==after[key],key
assert len(after['documents'])==372
for a,b in zip(before['documents'],after['documents'],strict=True):
    for key in ('index','sha256','scored_tokens','scored_bytes'): assert a[key]==b[key]
    assert math.isfinite(b['total_nats'])
assert math.isclose(sum(x['total_nats'] for x in after['documents']),after['total_nats'],abs_tol=1e-8)
result={'complete':True,'step':256,'tokens_seen':524288,'checkpoint':meta,'evaluation_sha256':hashlib.sha256(report_path.read_bytes()).hexdigest(),'finite_tensor_counts':counts,'skipped_nonfinite':0,'training_rows':len(rows),'cuda_rng_saved':True,'baseline_nats_per_token':before['nats_per_token'],'recovered_nats_per_token':after['nats_per_token'],'recovered_bits_per_byte':after['bits_per_byte'],'all_document_identities_and_denominators_match':True,'scope':'local checkpoint audit; remote durability not yet established','audit_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'torch':str(torch.__version__)}
out.write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
