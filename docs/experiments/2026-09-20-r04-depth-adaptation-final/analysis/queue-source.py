import json, subprocess, threading, time, traceback, hashlib, zipfile
import xml.etree.ElementTree as ET
assert 'depth_analysis_thread' not in globals()
assert depth_long_thread.is_alive() or depth_long_state['status']=='both_512_steps_and_depth_evaluations_complete'
depth_analysis_revision='7f74a23038eef534bd6e828503c3abaa5ac663db'
depth_analysis_repo=Path('/content/r04-depth-analysis-code-7f74a23')
depth_analysis_root=Path('/content/r04-depth-adaptation-analysis-v1')
assert not depth_analysis_repo.exists() and not depth_analysis_root.exists()
depth_analysis_root.mkdir()
depth_analysis_environment={**strict_gate_environment,'PYTHONPATH':str(depth_analysis_repo),'CUDA_VISIBLE_DEVICES':''}
depth_analysis_processes={}
depth_analysis_state={'status':'starting','analysis_revision':depth_analysis_revision,'training_revision':strict_gate_revision,'post_training_maximum_seconds':600,'wait_limit_seconds':5400,'scope':'CPU-only final checkpoint integrity and preregistered depth-policy screen. No additional training or inference.'}
depth_final_audit_source=r'''import sys, json, hashlib
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
'''
compile(depth_final_audit_source,'final-checkpoint-audit','exec')
(depth_analysis_root/'checkpoint-audit.py').write_text(depth_final_audit_source)
def depth_analysis_save():
    (depth_analysis_root/'queue.json').write_text(json.dumps(depth_analysis_state,indent=2)+chr(10))
def depth_analysis_run():
    started=time.monotonic()
    deadline=started+180
    try:
        def run(label,command,cwd=None):
            depth_analysis_state.update(status=label,command=command)
            with (depth_analysis_root/(label+'.txt')).open('x') as log:
                p=subprocess.Popen(command,cwd=cwd or depth_analysis_repo,env=depth_analysis_environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                depth_analysis_processes[label]=p
                depth_analysis_state['pid']=p.pid
                depth_analysis_save()
                try: status=p.wait(timeout=max(1,deadline-time.monotonic()))
                except subprocess.TimeoutExpired:
                    p.terminate()
                    try: p.wait(timeout=20)
                    except subprocess.TimeoutExpired: p.kill(); p.wait()
                    raise
                assert status==0,(label,status)
        run('fetch',['git','fetch','origin','claude/prophet-v03-memory-context'],recovery_repo)
        run('worktree',['git','worktree','add','--detach',str(depth_analysis_repo),depth_analysis_revision],recovery_repo)
        run('screen-tests',[sys.executable,'-m','pytest','tests/test_depth_summary.py','-q','--junitxml',str(depth_analysis_root/'screen-tests.xml')])
        suites=list(ET.parse(depth_analysis_root/'screen-tests.xml').getroot().iter('testsuite'))
        assert sum(int(s.attrib['tests']) for s in suites)==14
        assert all(int(s.attrib[k])==0 for s in suites for k in ['failures','errors','skipped'])
        depth_analysis_state.update(status='waiting_for_both_final_training_reports',screen_tests_passed=14)
        depth_analysis_save()
        depth_long_thread.join(timeout=5400)
        assert not depth_long_thread.is_alive() and all(p.poll()==0 for p in depth_long_processes.values())
        assert depth_long_state['status']=='both_512_steps_and_depth_evaluations_complete'
        assert depth_long_state['elapsed_seconds']<5400
        analysis_started=time.monotonic()
        deadline=analysis_started+600
        fixed=Path(depth_long_state['outputs']['fixed4'])
        variable=Path(depth_long_state['outputs']['uniform2to6'])
        run('checkpoint-audit',[sys.executable,'-u',str(depth_analysis_root/'checkpoint-audit.py'),str(depth_analysis_root/'final-state-audits.json'),str(fixed),str(variable)])
        run('depth-screen',[sys.executable,'-u',str(depth_analysis_repo/'scripts/summarize_depth_adaptation.py'),'--fixed',str(fixed/'evaluation-step-000512.json'),'--variable',str(variable/'evaluation-step-000512.json'),'--out',str(depth_analysis_root/'summary.json')])
        summary=json.loads((depth_analysis_root/'summary.json').read_text())
        depth_analysis_state.update(status='complete',seed0_screen_passed=summary['seed0_screen_passed'],checks=summary['checks'],analysis_seconds=time.monotonic()-analysis_started,elapsed_seconds_including_wait=time.monotonic()-started)
        depth_analysis_save()
        export=Path('/content/prophet-r04-depth-adaptation-final-evidence.zip')
        assert not export.exists()
        manifest={'analysis_revision':depth_analysis_revision,'training_revision':strict_gate_revision,'restart_gate_archive_sha256':depth_long_state['restart_gate_archive_sha256'],'training_processes':{k:{'pid':p.pid,'returncode':p.poll()} for k,p in depth_long_processes.items()},'analysis_processes':{k:{'pid':p.pid,'returncode':p.poll()} for k,p in depth_analysis_processes.items()},'files':{},'scope':'Final and prefix reports, training logs, launch sources, exact checkpoint metadata and CPU integrity audit; no corpus text or weight files.'}
        with zipfile.ZipFile(export,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for label,folder in [('training-queue',depth_long_root),('analysis',depth_analysis_root),('fixed4',fixed),('uniform2to6',variable)]:
                for p in sorted(folder.rglob('*')):
                    if p.is_file() and p.suffix in ['.json','.jsonl','.txt','.xml','.py']:
                        name=label+'/'+str(p.relative_to(folder))
                        raw=p.read_bytes()
                        manifest['files'][name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
                        z.writestr(name,raw)
            z.writestr('export-manifest.json',json.dumps(manifest,indent=2)+chr(10))
        depth_analysis_state['export']={'path':str(export),'bytes':export.stat().st_size,'sha256':hashlib.sha256(export.read_bytes()).hexdigest(),'files':len(manifest['files'])}
        depth_analysis_save()
    except BaseException as exc:
        depth_analysis_state.update(status='stopped_requires_inspection',error=repr(exc),elapsed_seconds_including_wait=time.monotonic()-started)
        (depth_analysis_root/'error.txt').write_text(traceback.format_exc())
        depth_analysis_save()
depth_analysis_thread=threading.Thread(target=depth_analysis_run,daemon=True)
depth_analysis_save()
depth_analysis_thread.start()
print('DEPTH_ANALYSIS_PREPARED',depth_analysis_state,flush=True)
