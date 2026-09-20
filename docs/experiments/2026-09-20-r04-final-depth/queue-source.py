import json, subprocess, threading, time, traceback
assert not final_chain_thread.is_alive()
assert all(p.poll() == 0 for p in final_processes.values())
assert all(p.poll() == 0 for p in covariance_processes.values())
assert all(p.poll() == 0 for p in recovery_pilot_processes.values())
assert 'final_depth_thread' not in globals()
final_depth_root = Path('/content/r04-final-depth-v1')
assert not final_depth_root.exists()
final_depth_root.mkdir()
final_depth_command = [sys.executable,'-u',str(covariance_repo/'scripts/eval_r04_depth.py'),'--model-repo',str(repo),'--run',str(final_stage_root/'loop-seed0'),'--corpus',str(corpus),'--expected-step','4096','--depths','4','1','2','6','8','--out',str(final_depth_root/'report.json')]
final_depth_state = {'status':'starting','command':final_depth_command,'maximum_seconds':600,'training_modified':False}
def final_depth_run():
    global final_depth_process
    started=time.monotonic()
    try:
        with (final_depth_root/'evaluation.txt').open('x') as log:
            final_depth_process=subprocess.Popen(final_depth_command,cwd=repo,env=chain_environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            final_depth_state.update(status='running',pid=final_depth_process.pid)
            (final_depth_root/'queue.json').write_text(json.dumps(final_depth_state,indent=2)+chr(10))
            try:
                code=final_depth_process.wait(timeout=600)
            except subprocess.TimeoutExpired:
                final_depth_process.terminate()
                try: final_depth_process.wait(timeout=20)
                except subprocess.TimeoutExpired: final_depth_process.kill(); final_depth_process.wait()
                raise TimeoutError('Final depth evaluation exceeded 600 seconds')
            assert code==0,code
        report=json.loads((final_depth_root/'report.json').read_text())
        assert report['complete'] and len(report['results'])==5
        final_depth_state.update(status='complete',elapsed_seconds=time.monotonic()-started)
    except BaseException as exc:
        final_depth_state.update(status='stopped_requires_inspection',error=repr(exc),elapsed_seconds=time.monotonic()-started)
        (final_depth_root/'error.txt').write_text(traceback.format_exc())
    (final_depth_root/'queue.json').write_text(json.dumps(final_depth_state,indent=2)+chr(10))
final_depth_thread=threading.Thread(target=final_depth_run,daemon=True)
final_depth_thread.start()
print('FINAL_DEPTH_STARTED',final_depth_state,flush=True)
