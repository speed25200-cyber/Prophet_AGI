import json, time, threading, traceback, subprocess, hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
assert 'covariance_thread' not in globals()
assert arc_eval_state['status'] == 'all_five_complete'
assert all(p.poll() == 0 for p in arc_eval_processes.values())
assert all(p.poll() == 0 for p in recovery_pilot_processes.values())
assert all(p.poll() == 0 for p in recovery_gpu_cache_processes.values())
covariance_revision = '90c970b5520a6b6c1da3d7bd78d5d7fc2b7567f6'
covariance_repo = recovery_root/'shared-fit-90c970b'
covariance_root = recovery_root/'activation-fit-v1'
assert not covariance_root.exists() and not covariance_repo.exists()
covariance_root.mkdir()
covariance_state = {'status':'starting','revision':covariance_revision,'maximum_seconds':1800}
covariance_processes = {}
covariance_environment = {**recovery_gpu_environment,'PYTHONPATH':str(covariance_repo),'CUBLAS_WORKSPACE_CONFIG':':4096:8'}
def covariance_save():
    (covariance_root/'queue.json').write_text(json.dumps(covariance_state,indent=2)+chr(10))
def covariance_run():
    started = time.monotonic()
    deadline = started+1800
    try:
        def run(name, command, cwd):
            covariance_state.update(status=name,command=command)
            with (covariance_root/(name+'.txt')).open('x') as stream:
                process = subprocess.Popen(command,cwd=cwd,env=covariance_environment,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                covariance_processes[name] = process
                covariance_state.update(pid=process.pid)
                covariance_save()
                try:
                    status = process.wait(timeout=max(1,deadline-time.monotonic()))
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try: process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait()
                    raise TimeoutError('Calibration queue exceeded its 1800-second ceiling')
                assert status == 0,(name,status)
        run('fetch',['git','fetch','origin','claude/prophet-v03-memory-context'],recovery_repo)
        run('worktree',['git','worktree','add','--detach',str(covariance_repo),covariance_revision],recovery_repo)
        run('solver-tests',[str(recovery_python),'-m','pytest','tests/test_shared_fit.py','-q','--junitxml',str(covariance_root/'solver-tests.xml')],covariance_repo)
        suites = ET.parse(covariance_root/'solver-tests.xml').getroot().iter('testsuite')
        suites = list(suites)
        assert sum(int(s.attrib['tests']) for s in suites) == 5
        assert all(int(s.attrib[k]) == 0 for s in suites for k in ['failures','errors','skipped'])
        covariance_state['actual_cpu_cuda_solver_tests_passed'] = 5
        command = [str(recovery_python),'-u',str(covariance_repo/'scripts/probe_qwen_covariance.py'),'--source',str(recovery_root/'source'),'--train',str(recovery_root/'data/train.jsonl'),'--validation',str(recovery_root/'data/validation.jsonl'),'--out',str(covariance_root/'fit'),'--device','cuda']
        run('calibration-first-document',command+['--max-documents-per-session','1'],covariance_repo)
        first = json.loads((covariance_root/'fit/report.json').read_text())
        assert not first['complete'] and first['next_document'] == 1
        covariance_state['first_document_checkpoint_created'] = True
        (covariance_root/'first-document-report.json').write_text(json.dumps(first,indent=2)+chr(10))
        run('calibration-resumed',command+['--max-documents-per-session','64'],covariance_repo)
        final = json.loads((covariance_root/'fit/report.json').read_text())
        assert final['complete'] and final['resumed_from'] == 1 and final['next_document'] == 64 and final['fit_count'] == 70
        covariance_state.update(status='complete',results={k:v['nats_per_token'] for k,v in final['results'].items()},resumed_from=1,elapsed_seconds=time.monotonic()-started)
    except BaseException as exc:
        covariance_state.update(status='stopped_requires_inspection',error=repr(exc),elapsed_seconds=time.monotonic()-started)
        (covariance_root/'error.txt').write_text(traceback.format_exc())
    covariance_save()
covariance_thread = threading.Thread(target=covariance_run,daemon=True)
covariance_save()
covariance_thread.start()
print('COVARIANCE_QUEUE_STARTED',covariance_state,flush=True)
