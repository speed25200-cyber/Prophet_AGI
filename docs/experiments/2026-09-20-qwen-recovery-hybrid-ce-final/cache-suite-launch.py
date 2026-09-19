cache_suite_revision='ed72c867d5f90fc614e82aca1866a9aad5f3dcfe'
cache_suite_repo=recovery_root/'checkpoint-analysis-ed72c86'
cache_suite_root=recovery_root/'recovered-cache-final-suite'
assert not cache_suite_repo.exists() and not cache_suite_root.exists()
assert recovery_pilot_thread.is_alive()
cache_suite_root.mkdir()
cache_suite_state={'status':'waiting_for_audited_first_final','revision':cache_suite_revision,'documents':4,'lengths':[128,512],'atol':1e-4,'rtol':1e-4}
cache_suite_process=None
def cache_suite_save():
    pilot_write(cache_suite_root/'queue.json',cache_suite_state)
def cache_suite_work():
    global cache_suite_process
    try:
        deadline=time.monotonic()+5400
        while 'hybrid-ce' not in recovery_pilot_state['results']:
            assert recovery_pilot_thread.is_alive() and recovery_pilot_state['status']!='failed', recovery_pilot_state
            assert time.monotonic()<deadline, 'Existing first run still pending; inspect without restarting'
            time.sleep(10)
        assert recovery_pilot_state['results']['hybrid-ce']['step']==2048
        subprocess.run(['git','fetch','origin',cache_suite_revision],cwd=recovery_gpu_repo,check=True,capture_output=True)
        subprocess.run(['git','worktree','add','--detach',str(cache_suite_repo),cache_suite_revision],cwd=recovery_gpu_repo,check=True,capture_output=True)
        assert diagnostic_head(recovery_gpu_repo)==recovery_run_protocol['revision']
        assert diagnostic_head(recovery_repo)=='8ad19d3c60dafd85fc67caf759981f1d765c3405'
        command=[str(recovery_python),str(cache_suite_repo/'scripts/audit_recovery_cache_suite.py'),'--run',str(recovery_first_run),'--step','2048','--source',str(recovery_root/'source'),'--validation',str(recovery_root/'data/validation.jsonl'),'--out',str(cache_suite_root/'chunk64.json'),'--documents','4','--lengths','128','512']
        cache_suite_state.update(status='running',command=command,checkpoint=recovery_pilot_state['results']['hybrid-ce']['checkpoint'])
        with (cache_suite_root/'chunk64.txt').open('x') as log:
            cache_suite_process=subprocess.Popen(command,cwd=cache_suite_repo,env=recovery_setup_environment,stdout=log,stderr=subprocess.STDOUT)
            cache_suite_state['pid']=cache_suite_process.pid
            cache_suite_save()
            try: code=cache_suite_process.wait(timeout=3600)
            except subprocess.TimeoutExpired:
                cache_suite_process.terminate()
                try: cache_suite_process.wait(timeout=60)
                except subprocess.TimeoutExpired: cache_suite_process.kill();cache_suite_process.wait(timeout=30)
                raise RuntimeError('CPU cache suite exceeded one hour')
        report=json.loads((cache_suite_root/'chunk64.json').read_bytes())
        assert report['complete'] and len(report['cases'])==8
        assert report['recovery_checkpoint_audit']['checkpoint']==cache_suite_state['checkpoint']
        assert (code==0 and report['passed']) or (code==1 and not report['passed'])
        cache_suite_state.update(status='passed' if report['passed'] else 'numerical_failure_report_preserved',exit_code=code,report_sha256=pilot_hash(cache_suite_root/'chunk64.json'))
    except Exception:
        cache_suite_state.update(status='error',error=traceback.format_exc())
    finally: cache_suite_save()
cache_suite_save()
cache_suite_thread=threading.Thread(target=cache_suite_work,name='final-cpu-cache-suite',daemon=True)
cache_suite_thread.start()
print('FINAL_CACHE_SUITE',cache_suite_state,flush=True)
print('FIRST_LIVE',recovery_second_process.poll(),'PILOT',recovery_pilot_state['status'],flush=True)