import json, subprocess, threading, time, traceback, hashlib, zipfile
import xml.etree.ElementTree as ET
assert 'recurrence_thread' not in globals()
recurrence_revision='2e4d25ecd521980ee71607b2d96d3ed13b224210'
recurrence_repo=Path('/content/r04-recurrence-observation-code-2e4d25e')
recurrence_root=Path('/content/r04-recurrence-observation-v1')
assert not recurrence_repo.exists() and not recurrence_root.exists()
recurrence_root.mkdir()
recurrence_environment={**strict_gate_environment,'PYTHONPATH':str(recurrence_repo)}
recurrence_processes={}
recurrence_state={'status':'starting','revision':recurrence_revision,'model_revision':strict_gate_revision,'maximum_pair_seconds':600,'wait_limit_seconds':6000,'scope':'Inference-only fixed first-sixteen-document internal-state observations after both final reports and integrity audit. No training or primary-screen changes.'}
def recurrence_save():
    (recurrence_root/'queue.json').write_text(json.dumps(recurrence_state,indent=2)+chr(10))
def recurrence_run():
    started=time.monotonic()
    deadline=started+180
    try:
        def run(label,command,cwd=None):
            recurrence_state.update(status=label,command=command)
            with (recurrence_root/(label+'.txt')).open('x') as log:
                p=subprocess.Popen(command,cwd=cwd or recurrence_repo,env=recurrence_environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                recurrence_processes[label]=p
                recurrence_state['pid']=p.pid
                recurrence_save()
                try: status=p.wait(timeout=max(1,deadline-time.monotonic()))
                except subprocess.TimeoutExpired:
                    p.terminate()
                    try: p.wait(timeout=20)
                    except subprocess.TimeoutExpired: p.kill(); p.wait()
                    raise
                assert status==0,(label,status)
        run('fetch',['git','fetch','origin','claude/prophet-v03-memory-context'],recovery_repo)
        run('worktree',['git','worktree','add','--detach',str(recurrence_repo),recurrence_revision],recovery_repo)
        run('tests',[sys.executable,'-m','pytest','tests/test_recurrence_observation.py','-q','--junitxml',str(recurrence_root/'tests.xml')])
        suites=list(ET.parse(recurrence_root/'tests.xml').getroot().iter('testsuite'))
        assert sum(int(s.attrib['tests']) for s in suites)==8
        assert all(int(s.attrib[k])==0 for s in suites for k in ['failures','errors','skipped'])
        (recurrence_root/'executed-driver.py').write_bytes((recurrence_repo/'scripts/diagnose_r04_recurrence.py').read_bytes())
        recurrence_state.update(status='waiting_for_final_evidence',tests_passed=8,setup_seconds=time.monotonic()-started)
        recurrence_save()
        depth_analysis_thread.join(timeout=6000)
        assert not depth_analysis_thread.is_alive() and not depth_long_thread.is_alive()
        assert all(p.poll()==0 for p in depth_analysis_processes.values()) and all(p.poll()==0 for p in depth_long_processes.values())
        assert depth_analysis_state['status']=='complete' and 'export' in depth_analysis_state
        assert depth_long_state['status']=='both_512_steps_and_depth_evaluations_complete'
        recurrence_state['final_evidence_sha256']=depth_analysis_state['export']['sha256']
        pair_started=time.monotonic()
        deadline=pair_started+600
        results={}
        for arm in ['fixed4','uniform2to6']:
            output=recurrence_root/(arm+'.json')
            run(arm,[sys.executable,'-u',str(recurrence_repo/'scripts/diagnose_r04_recurrence.py'),'--model-repo','/content/r04-depth-adaptation-code-f5a7d71','--run',depth_long_state['outputs'][arm],'--corpus',str(corpus),'--out',str(output)])
            result=json.loads(output.read_text())
            assert result['complete'] and result['identity']['arm']==arm and len(result['documents'])==16
            assert result['driver_sha256']==hashlib.sha256((recurrence_root/'executed-driver.py').read_bytes()).hexdigest()
            assert all(d['observer_hidden_equal'] and len(d['loops'])==8 for d in result['documents'])
            results[arm]=result
        assert [(d['index'],d['sha256'],d['input_ids_sha256']) for d in results['fixed4']['documents']]==[(d['index'],d['sha256'],d['input_ids_sha256']) for d in results['uniform2to6']['documents']]
        recurrence_state.update(status='complete',pair_seconds=time.monotonic()-pair_started,elapsed_seconds_including_wait=time.monotonic()-started)
        assert recurrence_state['pair_seconds']<600
        recurrence_save()
        export=Path('/content/prophet-r04-recurrence-observation.zip')
        assert not export.exists()
        manifest={'revision':recurrence_revision,'model_revision':strict_gate_revision,'processes':{k:{'pid':p.pid,'returncode':p.poll()} for k,p in recurrence_processes.items()},'files':{},'scope':'Queue, source, test and scalar observation records only; no weights or activation arrays.'}
        with zipfile.ZipFile(export,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for p in sorted(recurrence_root.rglob('*')):
                if p.is_file() and p.suffix in ['.json','.txt','.xml','.py']:
                    name=str(p.relative_to(recurrence_root))
                    raw=p.read_bytes()
                    manifest['files'][name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
                    z.writestr(name,raw)
            z.writestr('export-manifest.json',json.dumps(manifest,indent=2)+chr(10))
        recurrence_state['export']={'path':str(export),'bytes':export.stat().st_size,'sha256':hashlib.sha256(export.read_bytes()).hexdigest(),'files':len(manifest['files'])}
        recurrence_save()
    except BaseException as exc:
        recurrence_state.update(status='stopped_requires_inspection',error=repr(exc),elapsed_seconds_including_wait=time.monotonic()-started)
        (recurrence_root/'error.txt').write_text(traceback.format_exc())
        recurrence_save()
recurrence_thread=threading.Thread(target=recurrence_run,daemon=True)
recurrence_save()
recurrence_thread.start()
print('RECURRENCE_PREPARED',recurrence_state,flush=True)
