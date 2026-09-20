assert 'arc_eval_thread' not in globals(), 'Preserve the existing capability queue'
arc_eval_revision='6d36dced81bfd3378279bcabe9cba9df5304f0be'
arc_eval_repo=recovery_root/'arc-eval-6d36dce'
arc_eval_root=recovery_root/'arc-evaluation-v1'
assert not arc_eval_repo.exists() and not arc_eval_root.exists()
assert recovery_gpu_cache_thread.is_alive()
arc_eval_root.mkdir()
arc_eval_state={'status':'preparing_cpu_only','revision':arc_eval_revision,'results':{}}
arc_eval_processes={}
def arc_save():
    temporary=arc_eval_root/'queue.tmp'
    temporary.write_text(json.dumps(arc_eval_state,indent=2)+'\n')
    temporary.replace(arc_eval_root/'queue.json')
def arc_run(key,command,environment,timeout):
    arc_eval_state.update(status='running_'+key,command=[str(x) for x in command])
    with (arc_eval_root/(key+'.txt')).open('x') as log:
        process=subprocess.Popen([str(x) for x in command],cwd=arc_eval_repo,env=environment,stdout=log,stderr=subprocess.STDOUT)
        arc_eval_processes[key]=process
        arc_eval_state['pid']=process.pid
        arc_save()
        try: code=process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.terminate()
            try: process.wait(timeout=60)
            except subprocess.TimeoutExpired: process.kill();process.wait(timeout=30)
            raise RuntimeError(key+' exceeded its bounded runtime')
    assert code==0,(key,code)
def run_arc_evaluations():
    try:
        with (arc_eval_root/'setup.txt').open('x') as log:
            subprocess.run(['git','fetch','origin','claude/prophet-v03-memory-context'],cwd=recovery_repo,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
            subprocess.run(['git','worktree','add','--detach',str(arc_eval_repo),arc_eval_revision],cwd=recovery_repo,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
        assert diagnostic_head(arc_eval_repo)==arc_eval_revision
        assert diagnostic_head(recovery_gpu_repo)==recovery_run_protocol['revision']
        expected=json.loads((arc_eval_repo/'docs/experiments/2026-09-20-arc-recovery-protocol/manifest.json').read_text())
        arc_run('scorer-tests',[recovery_python,'-m','pytest','tests/test_choice_eval.py','-q','--junitxml',arc_eval_root/'scorer-tests.xml'],recovery_setup_environment,180)
        import xml.etree.ElementTree as arc_et
        suites=arc_et.parse(arc_eval_root/'scorer-tests.xml').getroot().findall('.//testsuite')
        assert sum(int(s.get('tests',0)) for s in suites)==12
        assert sum(int(s.get(key,0)) for s in suites for key in ['failures','errors','skipped'])==0
        arc_run('prepare',[recovery_python,arc_eval_repo/'scripts/prepare_arc_recovery_eval.py','--source',recovery_root/'source','--out',arc_eval_root/'items'],recovery_setup_environment,600)
        actual=json.loads((arc_eval_root/'items/manifest.json').read_text())
        assert actual==expected
        assert recovery_snapshot_digest(arc_eval_root/'items/items.jsonl')==expected['items_sha256']
        arc_eval_state.update(status='waiting_for_training_and_gpu_cache',input_manifest=actual,scorer_tests_passed=12)
        arc_save()
        deadline=time.monotonic()+5*3600
        while recovery_gpu_cache_thread.is_alive():
            if time.monotonic()>deadline: raise TimeoutError('GPU queue did not finish within five hours')
            recovery_gpu_cache_thread.join(timeout=30)
        assert recovery_gpu_cache_state['status'] in ['all_passed','numerical_failures_preserved'],recovery_gpu_cache_state
        assert not recovery_pilot_thread.is_alive() and all(p.poll()==0 for p in recovery_pilot_processes.values())
        assert all(p.poll() is not None for p in recovery_gpu_cache_processes.values())
        assert diagnostic_head(recovery_gpu_repo)==recovery_run_protocol['revision']
        arc_eval_state['preceding_cache_status']=recovery_gpu_cache_state['status']
        for arm in ['donor','hybrid-ce','attention-ce','hybrid-kl','attention-kl']:
            command=[recovery_python,arc_eval_repo/'scripts/eval_arc_recovery.py','--source',recovery_root/'source','--items',arc_eval_root/'items','--out',arc_eval_root/(arm+'.json'),'--device','cuda','--arm','donor' if arm=='donor' else 'recovered']
            if arm!='donor': command+=['--run',recovery_run_root/(arm+'-seed0'),'--step','2048']
            arc_run(arm,command,recovery_gpu_environment,1800)
            report=json.loads((arc_eval_root/(arm+'.json')).read_text())
            assert report['complete'] and report['manifest']==expected and report['evaluation']['rows']==2376
            assert report['runtime']['device']=='cuda' and report['runtime']['precision']=='float32'
            assert report['runtime']['allow_tf32_matmul'] is False and report['runtime']['allow_tf32_cudnn'] is False
            if arm!='donor': assert report['checkpoint_audit']['checkpoint']==recovery_pilot_state['results'][arm]['checkpoint']
            arc_eval_state['results'][arm]={'report_sha256':recovery_snapshot_digest(arc_eval_root/(arm+'.json')),**{key:report['evaluation'][key] for key in ['accuracy','accuracy_character_normalized','gold_answer_bits_per_byte','uniform_choice_chance']}}
            arc_save()
        arc_eval_state['status']='all_five_complete'
    except Exception:
        arc_eval_state.update(status='error',error=traceback.format_exc())
    finally: arc_save()
arc_eval_thread=threading.Thread(target=run_arc_evaluations,name='arc-after-training-and-cache',daemon=True)
arc_eval_thread.start()
print('ARC_QUEUE_STARTED',arc_eval_state,flush=True)
print('TRAINING_LIVE',recovery_pilot_state['status'],{k:(p.pid,p.poll()) for k,p in recovery_pilot_processes.items()},flush=True)
print('TRAINING_TAIL',(recovery_run_root/'hybrid-kl-queued-segment1.txt').read_text()[-350:],flush=True)