recovery_gpu_cache_revision='bfce7c522f4de9e2cba726135bbcbe2cb6adcacc'
recovery_gpu_cache_repo=recovery_root/'gpu-cache-bfce7c5'
recovery_gpu_cache_root=recovery_root/'gpu-recovered-cache-v1'
assert not recovery_gpu_cache_repo.exists() and not recovery_gpu_cache_root.exists()
assert recovery_pilot_thread.is_alive() and recovery_pilot_state['status']!='failed'
recovery_gpu_cache_root.mkdir()
recovery_gpu_cache_state={'status':'waiting_for_all_training_to_finish','revision':recovery_gpu_cache_revision,'results':{}}
recovery_gpu_cache_processes={}
def gpu_cache_save(): pilot_write(recovery_gpu_cache_root/'queue.json',recovery_gpu_cache_state)
def gpu_cache_run(key,command,timeout):
    with (recovery_gpu_cache_root/(key+'.txt')).open('x') as log:
        process=subprocess.Popen(command,cwd=recovery_gpu_cache_repo,env=recovery_gpu_environment,stdout=log,stderr=subprocess.STDOUT)
        recovery_gpu_cache_processes[key]=process
        recovery_gpu_cache_state.update(status='running_'+key,pid=process.pid,command=command)
        gpu_cache_save()
        try: return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.terminate()
            try: process.wait(timeout=60)
            except subprocess.TimeoutExpired: process.kill();process.wait(timeout=30)
            raise RuntimeError('GPU cache process deadline: '+key)
def run_all_gpu_cache_checks():
    try:
        for arm in recovery_run_protocol['arms']: wait_for_pilot_arm(arm)
        recovery_pilot_thread.join(timeout=60)
        assert not recovery_pilot_thread.is_alive() and recovery_pilot_state['status']=='four_arms_complete_local_audits_passed_remote_persistence_pending'
        assert all(p.poll()==0 for p in recovery_pilot_processes.values())
        subprocess.run(['git','fetch','origin',recovery_gpu_cache_revision],cwd=recovery_repo,check=True,capture_output=True)
        subprocess.run(['git','worktree','add','--detach',str(recovery_gpu_cache_repo),recovery_gpu_cache_revision],cwd=recovery_repo,check=True,capture_output=True)
        assert diagnostic_head(recovery_gpu_repo)==recovery_run_protocol['revision']
        command=[str(recovery_python),'-m','pytest','tests/test_recovery_cli.py','-q','-k','recovery_cache_suite_checks_longer_prefix','--junitxml',str(recovery_gpu_cache_root/'cuda-suite-tests.xml')]
        assert gpu_cache_run('cuda-suite-tests',command,600)==0
        import xml.etree.ElementTree as ET
        suites=list(ET.parse(recovery_gpu_cache_root/'cuda-suite-tests.xml').getroot().iter('testsuite'))
        assert sum(int(s.get('tests',0)) for s in suites)==4
        assert sum(int(s.get(key,0)) for s in suites for key in ['failures','errors','skipped'])==0
        recovery_gpu_cache_state['unit_tests']={'passed':4,'cuda_cases':2,'skipped':0}
        reference=json.loads((cache_suite_root/'chunk64.json').read_bytes())
        inputs=[(c['document_sha256'],c['length'],c['input_ids_sha256']) for c in reference['cases']]
        for arm in recovery_run_protocol['arms']:
            output=recovery_gpu_cache_root/(arm+'.json')
            command=[str(recovery_python),str(recovery_gpu_cache_repo/'scripts/audit_recovery_cache_suite.py'),'--run',str(recovery_run_root/(arm+'-seed0')),'--step','2048','--source',str(recovery_root/'source'),'--validation',str(recovery_root/'data/validation.jsonl'),'--out',str(output),'--documents','4','--lengths','128','512','--device','cuda']
            code=gpu_cache_run(arm,command,3600)
            report=json.loads(output.read_bytes())
            assert report['complete'] and report['device']=='cuda' and report['precision']=='float32'
            assert report['allow_tf32_matmul'] is False and report['allow_tf32_cudnn'] is False
            assert report['gdn_scan']==('fla_chunk32' if arm.startswith('hybrid') else None)
            assert report['recovery_checkpoint_audit']['checkpoint']==recovery_pilot_state['results'][arm]['checkpoint']
            assert [(c['document_sha256'],c['length'],c['input_ids_sha256']) for c in report['cases']]==inputs
            assert (code==0 and report['passed']) or (code==1 and not report['passed'])
            recovery_gpu_cache_state['results'][arm]={'passed':report['passed'],'exit_code':code,'sha256':pilot_hash(output),'checkpoint':report['recovery_checkpoint_audit']['checkpoint']}
            gpu_cache_save()
        recovery_gpu_cache_state['status']='all_passed' if all(x['passed'] for x in recovery_gpu_cache_state['results'].values()) else 'numerical_failures_preserved'
    except Exception:
        recovery_gpu_cache_state.update(status='error',error=traceback.format_exc())
    finally: gpu_cache_save()
gpu_cache_save()
recovery_gpu_cache_thread=threading.Thread(target=run_all_gpu_cache_checks,name='post-pilot-gpu-cache-checks',daemon=True)
recovery_gpu_cache_thread.start()
print('POST_TRAINING_GPU_CHECKS_QUEUED',recovery_gpu_cache_state,flush=True)
print('CURRENT_TRAINING',recovery_pilot_state['status'],{k:(p.pid,p.poll()) for k,p in recovery_pilot_processes.items()},flush=True)
print('ATTENTION_TAIL',(recovery_run_root/'attention-ce-queued-segment1.txt').read_text()[-400:],flush=True)