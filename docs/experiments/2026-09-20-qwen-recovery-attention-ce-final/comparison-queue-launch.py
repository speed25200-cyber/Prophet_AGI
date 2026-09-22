recovery_comparison_revision='e3ec54676be00daa190e183a387dd5ee22d3e42f'
recovery_comparison_repo=recovery_root/'comparison-e3ec546'
recovery_comparison_root=recovery_root/'recovery-comparisons-v1'
assert not recovery_comparison_repo.exists() and not recovery_comparison_root.exists()
assert recovery_pilot_thread.is_alive() and recovery_pilot_state['status']!='failed'
assert not cache_suite_thread.is_alive() and cache_suite_process.poll()==0
assert all((fp32_root/('baseline-'+arm+'.json')).exists() for arm in ['donor','hybrid','attention'])
recovery_comparison_root.mkdir()
recovery_cache_all_state={'status':'waiting_for_attention_final','revision':cache_suite_revision,'results':{}}
recovery_cache_all_processes={}
recovery_analysis_state={'status':'waiting_for_all_four_audited_endpoints','revision':recovery_comparison_revision}
recovery_analysis_process=None
def cache_all_save(): pilot_write(recovery_comparison_root/'cache-queue.json',recovery_cache_all_state)
def analysis_save(): pilot_write(recovery_comparison_root/'analysis-queue.json',recovery_analysis_state)
def wait_for_pilot_arm(arm):
    deadline=time.monotonic()+18000
    while arm not in recovery_pilot_state['results']:
        assert recovery_pilot_thread.is_alive() and recovery_pilot_state['status']!='failed', recovery_pilot_state
        assert time.monotonic()<deadline, 'Pilot remains pending; inspect its live process before any restart'
        time.sleep(15)
    assert recovery_pilot_state['results'][arm]['step']==2048
def run_remaining_cache_suites():
    try:
        first=json.loads((cache_suite_root/'chunk64.json').read_bytes())
        reference_inputs=[(c['document_sha256'],c['length'],c['input_ids_sha256']) for c in first['cases']]
        recovery_cache_all_state['results']['hybrid-ce']={'passed':True,'report':str(cache_suite_root/'chunk64.json'),'sha256':pilot_hash(cache_suite_root/'chunk64.json'),'checkpoint':first['recovery_checkpoint_audit']['checkpoint'],'reused_existing_verified_report':True}
        cache_all_save()
        for arm in recovery_run_protocol['arms'][1:]:
            recovery_cache_all_state['status']='waiting_for_'+arm
            cache_all_save()
            wait_for_pilot_arm(arm)
            output=recovery_comparison_root/(arm+'-cache.json')
            command=[str(recovery_python),str(cache_suite_repo/'scripts/audit_recovery_cache_suite.py'),'--run',str(recovery_run_root/(arm+'-seed0')),'--step','2048','--source',str(recovery_root/'source'),'--validation',str(recovery_root/'data/validation.jsonl'),'--out',str(output),'--documents','4','--lengths','128','512']
            with (recovery_comparison_root/(arm+'-cache.txt')).open('x') as log:
                process=subprocess.Popen(command,cwd=cache_suite_repo,env=recovery_setup_environment,stdout=log,stderr=subprocess.STDOUT)
                recovery_cache_all_processes[arm]=process
                recovery_cache_all_state.update(status='running_'+arm,pid=process.pid,command=command)
                cache_all_save()
                try: code=process.wait(timeout=3600)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try: process.wait(timeout=60)
                    except subprocess.TimeoutExpired: process.kill();process.wait(timeout=30)
                    raise RuntimeError('Cache suite exceeded one hour: '+arm)
            report=json.loads(output.read_bytes())
            assert report['complete'] and len(report['cases'])==8
            assert report['recovery_checkpoint_audit']['checkpoint']==recovery_pilot_state['results'][arm]['checkpoint']
            assert [(c['document_sha256'],c['length'],c['input_ids_sha256']) for c in report['cases']]==reference_inputs
            assert (code==0 and report['passed']) or (code==1 and not report['passed'])
            recovery_cache_all_state['results'][arm]={'passed':report['passed'],'exit_code':code,'report':str(output),'sha256':pilot_hash(output),'checkpoint':report['recovery_checkpoint_audit']['checkpoint']}
            cache_all_save()
        recovery_cache_all_state['status']='all_passed' if all(x['passed'] for x in recovery_cache_all_state['results'].values()) else 'numerical_failures_preserved'
    except Exception:
        recovery_cache_all_state.update(status='error',error=traceback.format_exc())
    finally: cache_all_save()
def run_final_comparison():
    global recovery_analysis_process
    try:
        for arm in recovery_run_protocol['arms']: wait_for_pilot_arm(arm)
        recovery_pilot_thread.join(timeout=60)
        assert not recovery_pilot_thread.is_alive() and recovery_pilot_state['status']=='four_arms_complete_local_audits_passed_remote_persistence_pending'
        assert all(p.poll()==0 for p in recovery_pilot_processes.values())
        subprocess.run(['git','fetch','origin',recovery_comparison_revision],cwd=recovery_gpu_repo,check=True,capture_output=True)
        subprocess.run(['git','worktree','add','--detach',str(recovery_comparison_repo),recovery_comparison_revision],cwd=recovery_gpu_repo,check=True,capture_output=True)
        assert diagnostic_head(recovery_gpu_repo)==recovery_run_protocol['revision']
        output=recovery_comparison_root/'four-arm-comparison.json'
        command=[str(recovery_python),str(recovery_comparison_repo/'scripts/summarize_recovery_pilot.py'),'--root',str(recovery_run_root),'--baselines',str(fp32_root),'--out',str(output)]
        with (recovery_comparison_root/'four-arm-comparison.txt').open('x') as log:
            recovery_analysis_process=subprocess.Popen(command,cwd=recovery_comparison_repo,env=recovery_setup_environment,stdout=log,stderr=subprocess.STDOUT)
            recovery_analysis_state.update(status='running',pid=recovery_analysis_process.pid,command=command)
            analysis_save()
            try: code=recovery_analysis_process.wait(timeout=600)
            except subprocess.TimeoutExpired:
                recovery_analysis_process.terminate()
                try: recovery_analysis_process.wait(timeout=60)
                except subprocess.TimeoutExpired: recovery_analysis_process.kill();recovery_analysis_process.wait(timeout=30)
                raise RuntimeError('Comparison process timed out')
        assert code==0
        report=json.loads(output.read_bytes())
        assert report['complete'] and set(report['arms'])==set(recovery_run_protocol['arms'])
        recovery_analysis_state.update(status='complete',exit_code=0,sha256=pilot_hash(output),ce_ranking=report['ce_ranking'])
    except Exception:
        recovery_analysis_state.update(status='error',error=traceback.format_exc())
    finally: analysis_save()
cache_all_save();analysis_save()
recovery_cache_all_thread=threading.Thread(target=run_remaining_cache_suites,name='four-arm-cache-checks',daemon=True)
recovery_analysis_thread=threading.Thread(target=run_final_comparison,name='four-arm-matched-comparison',daemon=True)
recovery_cache_all_thread.start();recovery_analysis_thread.start()
print('CACHE_COMPARISONS_QUEUED',recovery_cache_all_state,flush=True)
print('QUALITY_COMPARISON_QUEUED',recovery_analysis_state,flush=True)
print('TRAINING_LIVE',{k:(p.pid,p.poll()) for k,p in recovery_pilot_processes.items()},flush=True)