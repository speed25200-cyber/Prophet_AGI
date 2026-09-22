import sys, json, hashlib
from pathlib import Path
from scripts.audit_r04_restart import load_run, compare_states
from scripts.run_r04_pilot import write_json
out=Path(sys.argv[1])
assert not out.exists()
result={'passed':True,'arms':{},'scope':'Restricted CPU reload of the published final files; byte hashes, checkpoint/report/contract/counters and all tensor finiteness verified. Self-comparison is used only to traverse tensors, not as evidence of cross-run equality or repeated inference.'}
shapes=None
for arm,folder in [('fixed4',Path(sys.argv[2])),('uniform2to6',Path(sys.argv[3]))]:
    state,report=load_run(folder,512)
    assert report['complete'] and report['tokens_seen']==8388608 and report['loader_step']==36864
    assert len(report['depth_history'])==512 and set(report['results'])=={'1','2','4','6','8'}
    contract=json.loads((folder/'adaptation.json').read_text())
    assert state['training_contract']==contract['training_contract']
    assert report['identity']==contract['identity'] and report['identity']['arm']==arm
    assert report['identity']['code']['revision']=='f5a7d71be1723324e36a7be2546c940a69de9fdf'
    counts=compare_states(state,state)
    assert counts=={'tensors':315,'tensor_elements':867172946}
    model_shapes={name:[list(tensor.shape),str(tensor.dtype)] for name,tensor in state['model'].items()}
    if shapes is None: shapes=model_shapes
    else: assert shapes==model_shapes
    result['arms'][arm]={'checkpoint':report['checkpoint'],'identity':report['identity'],'all_checkpoint_tensors_finite':True,'inspected':counts,'step':state['step'],'tokens_seen':state['tokens_seen'],'loader_step':state['loader']['step'],'depth_history':state['adaptation_depth_history'],'report_sha256':hashlib.sha256((folder/'evaluation-step-000512.json').read_bytes()).hexdigest()}
    del state
write_json(out,result)
print('FINAL_DEPTH_CHECKPOINTS_VERIFIED', {k:v['inspected'] for k,v in result['arms'].items()},flush=True)
