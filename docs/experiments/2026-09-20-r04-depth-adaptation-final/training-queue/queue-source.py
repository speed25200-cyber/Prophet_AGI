import json, subprocess, threading, time, traceback, shutil, hashlib
assert 'depth_long_thread' not in globals()
assert not strict_gate_thread.is_alive() and all(p.poll()==0 for p in strict_gate_processes.values())
assert strict_gate_state['status']=='both_real_shape_restart_gates_passed'
assert strict_gate_state['production_prefix']=='continuous'
assert gate_export_path.exists() and hashlib.sha256(gate_export_path.read_bytes()).hexdigest()=='e1935e51e12858e4d04099b34331501c824508f4f989fc667035fc3c54c02529'
assert shutil.disk_usage('/content').free>12_000_000_000
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=strict_gate_repo,text=True).strip()==strict_gate_revision
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=strict_gate_repo,text=True).strip()
for arm in ['fixed4','uniform2to6']:
    audit=json.loads((strict_gate_root/(arm+'-restart-audit.json')).read_text())
    assert audit['passed'] and audit['per_document_evaluation_equal'] and audit['step']==8
    assert audit['identity']['code']['revision']==strict_gate_revision
    assert audit['identity']['numerical_policy']['deterministic_algorithms']
    prefix=strict_gate_root/(arm+'-continuous')
    assert strict_gate_state['results'][arm]['production_prefix']==str(prefix)
    report=json.loads((prefix/'evaluation-step-000008.json').read_text())
    assert report['checkpoint']==audit['left_checkpoint'] and report['skipped_nonfinite']==0
    assert max(c['step'] for c in json.loads((prefix/'checkpoints/manifest.json').read_text())['checkpoints'])==8
depth_long_root=Path('/content/r04-depth-adaptation-long-v2')
assert not depth_long_root.exists()
depth_long_root.mkdir()
depth_long_processes={}
depth_long_state={'status':'starting','revision':strict_gate_revision,'maximum_seconds':5400,'restart_gate_archive_sha256':hashlib.sha256(gate_export_path.read_bytes()).hexdigest(),'outputs':{arm:str(strict_gate_root/(arm+'-continuous')) for arm in ['fixed4','uniform2to6']},'results':{},'scope':'Continue preselected continuous-eight prefixes to 64 then 512 steps. No quality-based early stopping or recipe selection. Gate diagnostics separately bounded at 1800 seconds.'}
def depth_long_save():
    (depth_long_root/'queue.json').write_text(json.dumps(depth_long_state,indent=2)+chr(10))
def depth_long_run():
    started=time.monotonic()
    deadline=started+5400
    try:
        for target,additional in [(64,56),(512,448)]:
            for arm in ['fixed4','uniform2to6']:
                output=Path(depth_long_state['outputs'][arm])
                label=arm+'-to'+str(target)
                command=[sys.executable,'-u',str(strict_gate_repo/'scripts/adapt_r04_depth.py'),'--parent-run',str(final_stage_root/'loop-seed0'),'--corpus',str(corpus),'--arm',arm,'--out',str(output),'--max-session-steps',str(additional),'--session-minutes','30']
                assert time.monotonic()<deadline
                depth_long_state.update(status=label,command=command)
                with (depth_long_root/(label+'.txt')).open('x') as log:
                    p=subprocess.Popen(command,cwd=strict_gate_repo,env=strict_gate_environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    depth_long_processes[label]=p
                    depth_long_state['pid']=p.pid
                    depth_long_save()
                    try: status=p.wait(timeout=max(1,deadline-time.monotonic()))
                    except subprocess.TimeoutExpired:
                        p.terminate()
                        try: p.wait(timeout=60)
                        except subprocess.TimeoutExpired: p.kill(); p.wait()
                        raise
                    assert status==0,(label,status)
                report=json.loads((output/('evaluation-step-'+str(target).zfill(6)+'.json')).read_text())
                assert report['step']==target and report['tokens_seen']==target*8*2048 and report['skipped_nonfinite']==0
                assert report['identity']['code']['revision']==strict_gate_revision
                assert report['identity']['numerical_policy']['deterministic_algorithms']
                assert report['complete']==(target==512)
                assert len(report['depth_history'])==target
                assert set(report['results'])==({'1','2','4','6','8'} if target==512 else {'4'})
                depth_long_state['results'][label]={'step':target,'tokens_seen':report['tokens_seen'],'depth_counts':report['depth_counts'],'complete':report['complete'],'report_sha256':hashlib.sha256((output/('evaluation-step-'+str(target).zfill(6)+'.json')).read_bytes()).hexdigest()}
                depth_long_save()
        depth_long_state['status']='both_512_steps_and_depth_evaluations_complete'
    except BaseException as exc:
        depth_long_state.update(status='stopped_requires_inspection',error=repr(exc))
        (depth_long_root/'error.txt').write_text(traceback.format_exc())
    depth_long_state['elapsed_seconds']=time.monotonic()-started
    depth_long_save()
depth_long_thread=threading.Thread(target=depth_long_run,daemon=True)
depth_long_save()
depth_long_thread.start()
print('DEPTH_LONG_QUEUE_STARTED',depth_long_state,flush=True)
