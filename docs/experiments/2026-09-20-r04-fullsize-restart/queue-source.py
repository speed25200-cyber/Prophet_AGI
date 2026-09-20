import json, subprocess, threading, time, traceback, shutil
import xml.etree.ElementTree as ET
assert 'strict_gate_thread' not in globals()
assert not adaptation_thread.is_alive() and all(p.poll() is not None for p in adaptation_processes.values())
assert not hash_probe_thread.is_alive() and all(p.poll()==0 for p in hash_probe_processes.values())
assert hash_probe_state['status']=='complete' and not hash_probe_state['comparison']['final_different']
assert shutil.disk_usage('/content').free>32_000_000_000
strict_gate_revision='f5a7d71be1723324e36a7be2546c940a69de9fdf'
strict_gate_repo=Path('/content/r04-depth-adaptation-code-f5a7d71')
strict_gate_root=Path('/content/r04-depth-strict-gate-v1')
assert not strict_gate_repo.exists() and not strict_gate_root.exists()
strict_gate_root.mkdir()
strict_gate_environment={**adaptation_environment,'PYTHONPATH':str(strict_gate_repo)}
strict_gate_processes={}
strict_gate_state={'status':'starting','revision':strict_gate_revision,'maximum_seconds':1800,'production_prefix':'continuous','long_training_queued':False,'results':{}}
def strict_gate_save():
    (strict_gate_root/'queue.json').write_text(json.dumps(strict_gate_state,indent=2)+chr(10))
def strict_gate_run():
    started=time.monotonic()
    deadline=started+1800
    try:
        def run(label,command,cwd=None):
            strict_gate_state.update(status=label,command=command)
            with (strict_gate_root/(label+'.txt')).open('x') as log:
                p=subprocess.Popen(command,cwd=cwd or strict_gate_repo,env=strict_gate_environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                strict_gate_processes[label]=p
                strict_gate_state['pid']=p.pid
                strict_gate_save()
                try: status=p.wait(timeout=max(1,deadline-time.monotonic()))
                except subprocess.TimeoutExpired:
                    p.terminate()
                    try: p.wait(timeout=60)
                    except subprocess.TimeoutExpired: p.kill(); p.wait()
                    raise
                assert status==0,(label,status)
        run('fetch',['git','fetch','origin','claude/prophet-v03-memory-context'],recovery_repo)
        run('worktree',['git','worktree','add','--detach',str(strict_gate_repo),strict_gate_revision],recovery_repo)
        run('cpu-cuda-tests',[sys.executable,'-m','pytest','tests/test_depth_adaptation.py','-q','--junitxml',str(strict_gate_root/'cpu-cuda-tests.xml')])
        suites=list(ET.parse(strict_gate_root/'cpu-cuda-tests.xml').getroot().iter('testsuite'))
        assert sum(int(s.attrib['tests']) for s in suites)==9
        assert all(int(s.attrib[k])==0 for s in suites for k in ['failures','errors','skipped'])
        assert sum(c.attrib['name'].endswith('cuda]') for s in suites for c in s.iter('testcase'))==2
        strict_gate_state['actual_cpu_cuda_tests_passed']=9
        for arm in ['fixed4','uniform2to6']:
            base=[sys.executable,'-u',str(strict_gate_repo/'scripts/adapt_r04_depth.py'),'--parent-run',str(final_stage_root/'loop-seed0'),'--corpus',str(corpus),'--arm',arm]
            run('preflight-'+arm,base+['--mode','preflight','--out',str(strict_gate_root/('preflight-'+arm))])
            report=json.loads((strict_gate_root/('preflight-'+arm)/'preflight.json').read_text())
            assert report['passed'] and report['actual_depths']==[6,6,6]
            assert report['identity']['numerical_policy']['deterministic_algorithms'] and report['identity']['code']['revision']==strict_gate_revision
        for arm in ['fixed4','uniform2to6']:
            base=[sys.executable,'-u',str(strict_gate_repo/'scripts/adapt_r04_depth.py'),'--parent-run',str(final_stage_root/'loop-seed0'),'--corpus',str(corpus),'--arm',arm,'--session-minutes','5']
            continuous=strict_gate_root/(arm+'-continuous')
            interrupted=strict_gate_root/(arm+'-interrupted')
            for label,output,count,target in [('continuous8',continuous,8,8),('interrupted1',interrupted,1,1),('resumed7',interrupted,7,8)]:
                run(arm+'-'+label,base+['--out',str(output),'--max-session-steps',str(count)])
                report=json.loads((output/('evaluation-step-'+str(target).zfill(6)+'.json')).read_text())
                assert report['step']==target and report['tokens_seen']==target*8*2048
                assert not report['complete'] and report['skipped_nonfinite']==0
                assert report['identity']['code']['revision']==strict_gate_revision
            run(arm+'-audit',[sys.executable,'-u',str(strict_gate_repo/'scripts/audit_r04_restart.py'),'--left',str(continuous),'--right',str(interrupted),'--expected-step','8','--out',str(strict_gate_root/(arm+'-restart-audit.json'))])
            audit=json.loads((strict_gate_root/(arm+'-restart-audit.json')).read_text())
            assert audit['passed'] and audit['per_document_evaluation_equal']
            if arm=='uniform2to6': assert len(set(audit['depth_history']))>1 and 6 in audit['depth_history']
            strict_gate_state['results'][arm]={'exact_restart':True,'depth_history':audit['depth_history'],'state_comparison':audit['state_comparison'],'production_prefix':str(continuous)}
            strict_gate_save()
        strict_gate_state['status']='both_real_shape_restart_gates_passed'
    except BaseException as exc:
        strict_gate_state.update(status='stopped_requires_inspection',error=repr(exc))
        (strict_gate_root/'error.txt').write_text(traceback.format_exc())
    strict_gate_state['elapsed_seconds']=time.monotonic()-started
    strict_gate_save()
strict_gate_thread=threading.Thread(target=strict_gate_run,daemon=True)
strict_gate_save()
strict_gate_thread.start()
print('STRICT_REAL_SHAPE_GATE_STARTED',strict_gate_state,flush=True)
