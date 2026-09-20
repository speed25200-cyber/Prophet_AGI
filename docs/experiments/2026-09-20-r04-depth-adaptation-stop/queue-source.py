import os, sys, json, time, threading, traceback, subprocess, shutil
from pathlib import Path
import xml.etree.ElementTree as ET
assert 'adaptation_thread' not in globals()
assert final_depth_process.poll()==0
assert all(p.poll()==0 for p in covariance_processes.values())
assert all(p.poll()==0 for p in final_processes.values())
assert all(p.poll()==0 for p in recovery_pilot_processes.values())
assert all(p.poll()==0 for p in recovery_gpu_cache_processes.values())
assert all(p.poll()==0 for p in arc_eval_processes.values())
adaptation_revision='4f5c56613f3b9b99485931945438cb86a8f70c7a'
adaptation_repo=Path('/content/r04-depth-adaptation-code-4f5c566')
adaptation_root=Path('/content/r04-depth-adaptation-v1')
assert not adaptation_repo.exists() and not adaptation_root.exists()
assert shutil.disk_usage('/content').free > 25*1024**3
adaptation_root.mkdir()
adaptation_environment={**chain_environment,'PYTHONPATH':str(adaptation_repo),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
adaptation_state={'status':'starting','revision':adaptation_revision,'maximum_seconds':5400,'results':{}}
adaptation_processes={}
def adaptation_save():
    (adaptation_root/'queue.json').write_text(json.dumps(adaptation_state,indent=2)+chr(10))
def adaptation_run():
    started=time.monotonic()
    deadline=started+5400
    try:
        def run(name, command, cwd=adaptation_repo):
            assert time.monotonic()<deadline
            adaptation_state.update(status=name,command=command)
            with (adaptation_root/(name+'.txt')).open('x') as log:
                process=subprocess.Popen(command,cwd=cwd,env=adaptation_environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                adaptation_processes[name]=process
                adaptation_state.update(pid=process.pid)
                adaptation_save()
                try: code=process.wait(timeout=max(1,deadline-time.monotonic()))
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try: process.wait(timeout=60)
                    except subprocess.TimeoutExpired: process.kill(); process.wait()
                    raise TimeoutError('Paired adaptation exceeded its 5400-second ceiling')
                assert code==0,(name,code)
        run('fetch',['git','fetch','origin','claude/prophet-v03-memory-context'],recovery_repo)
        run('worktree',['git','worktree','add','--detach',str(adaptation_repo),adaptation_revision],recovery_repo)
        run('cuda-restart-tests',[sys.executable,'-m','pytest','tests/test_depth_adaptation.py','-q','--junitxml',str(adaptation_root/'cuda-restart-tests.xml')])
        suites=list(ET.parse(adaptation_root/'cuda-restart-tests.xml').getroot().iter('testsuite'))
        cases=[c for suite in suites for c in suite.iter('testcase')]
        assert len(cases)==6 and sum('cuda]' in c.attrib['name'] for c in cases)==2
        assert all(int(s.attrib[k])==0 for s in suites for k in ['failures','errors','skipped'])
        adaptation_state.update(actual_cpu_cuda_tests_passed=6,actual_cuda_restart_cases_passed=2)
        for arm in ['fixed4','uniform2to6']:
            output=adaptation_root/('preflight-'+arm)
            command=[sys.executable,'-u',str(adaptation_repo/'scripts/adapt_r04_depth.py'),'--parent-run',str(final_stage_root/'loop-seed0'),'--corpus',str(corpus),'--arm',arm,'--mode','preflight','--out',str(output)]
            run('preflight-'+arm,command)
            result=json.loads((output/'preflight.json').read_text())
            assert result['passed'] and result['actual_depths']==[6,6,6]
            assert result['identity']['code']['revision']==adaptation_revision
            adaptation_state.setdefault('preflights',{})[arm]={'peak_allocated_bytes':result['peak_allocated_bytes'],'capacity_bytes':result['device_capacity_bytes'],'step_seconds':result['step_seconds']}
            adaptation_save()
        for count, target in [(64,64),(448,512)]:
            for arm in ['fixed4','uniform2to6']:
                output=adaptation_root/arm
                command=[sys.executable,'-u',str(adaptation_repo/'scripts/adapt_r04_depth.py'),'--parent-run',str(final_stage_root/'loop-seed0'),'--corpus',str(corpus),'--arm',arm,'--out',str(output),'--max-session-steps',str(count),'--session-minutes','20']
                run(arm+'-to-'+str(target),command)
                report=json.loads((output/('evaluation-step-'+str(target).zfill(6)+'.json')).read_text())
                assert report['step']==target and report['tokens_seen']==target*8*2048 and report['skipped_nonfinite']==0
                assert report['identity']['code']['revision']==adaptation_revision
                assert len(report['depth_history'])==target
                assert set(report['depth_history'])==({4} if arm=='fixed4' else {2,3,4,5,6})
                if target==512: assert report['complete'] and set(report['results'])=={'1','2','4','6','8'}
                adaptation_state['results'][arm]={'step':target,'depth_counts':report['depth_counts'],'ce':{k:v['nats_per_token'] for k,v in report['results'].items()},'checkpoint':report['checkpoint']}
                adaptation_save()
        adaptation_state.update(status='both_complete',elapsed_seconds=time.monotonic()-started)
    except BaseException as exc:
        adaptation_state.update(status='stopped_requires_inspection',error=repr(exc),elapsed_seconds=time.monotonic()-started)
        (adaptation_root/'error.txt').write_text(traceback.format_exc())
    adaptation_save()
adaptation_thread=threading.Thread(target=adaptation_run,daemon=True)
adaptation_save()
adaptation_thread.start()
print('DEPTH_ADAPTATION_QUEUE_STARTED',adaptation_state,flush=True)
