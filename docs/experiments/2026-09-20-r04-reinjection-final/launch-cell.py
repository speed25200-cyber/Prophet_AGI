(reinjection_gate_root/'launch-cell.py').write_text("import os, sys, json, time, signal, subprocess, threading, traceback, shutil, hashlib, xml.etree.ElementTree as ET\nfrom pathlib import Path\nassert all(p.poll() == 0 for group in [depth_long_processes, depth_analysis_processes, recurrence_processes] for p in group.values())\nreinjection_gate_revision = 'dce35abc439fdeeb8f1482b1219dee3cc242c267'\nreinjection_gate_root = Path('/content/r04-reinjection-gate-v1')\nreinjection_gate_repo = Path('/content/r04-reinjection-code-dce35ab')\nassert not reinjection_gate_root.exists() and not reinjection_gate_repo.exists()\nassert shutil.disk_usage('/content').free > 32_000_000_000\nreinjection_gate_root.mkdir()\nreinjection_gate_processes = {}\nreinjection_gate_state = {'status':'starting','revision':reinjection_gate_revision,'maximum_seconds':1800,'scope':'Initial actual-shape equality, CUDA tests, both k6 preflights and both separate-process 8 versus 1+7 restart gates. No long continuation.'}\nreinjection_gate_environment = {**os.environ,'PYTHONPATH':str(reinjection_gate_repo),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','TRITON_F32_DEFAULT':'tf32x3','CUBLAS_WORKSPACE_CONFIG':':4096:8'}\ndef reinjection_gate_queue():\n    began = time.monotonic()\n    def save():\n        reinjection_gate_state['elapsed_seconds'] = time.monotonic()-began\n        (reinjection_gate_root/'queue.json').write_text(json.dumps(reinjection_gate_state,indent=2)+chr(10))\n    def run(label, command, cwd=None):\n        remaining = 1800-(time.monotonic()-began)\n        assert remaining > 0, 'gate deadline exceeded'\n        reinjection_gate_state.update(status=label,command=command)\n        with (reinjection_gate_root/(label+'.txt')).open('w') as output:\n            p = subprocess.Popen(command,cwd=cwd,env=reinjection_gate_environment,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)\n            reinjection_gate_processes[label] = p\n            reinjection_gate_state['pid'] = p.pid\n            save()\n            try:\n                rc=p.wait(timeout=remaining)\n            except subprocess.TimeoutExpired:\n                os.killpg(p.pid,signal.SIGTERM)\n                try: p.wait(timeout=20)\n                except subprocess.TimeoutExpired:\n                    os.killpg(p.pid,signal.SIGKILL)\n                    p.wait()\n                raise RuntimeError('gate deadline exceeded; process terminated')\n        reinjection_gate_state.setdefault('processes',{})[label]={'pid':p.pid,'returncode':rc}\n        save()\n        assert rc == 0, label+' failed; inspect its log'\n    try:\n        checkout=Path('/content/prophet-recovery/repo')\n        assert checkout.exists()\n        run('fetch',['git','fetch','origin','claude/prophet-v03-memory-context'],str(checkout))\n        run('worktree',['git','worktree','add','--detach',str(reinjection_gate_repo),reinjection_gate_revision],str(checkout))\n        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=reinjection_gate_repo,text=True).strip()==reinjection_gate_revision\n        xml=reinjection_gate_root/'tests.xml'\n        run('tests',[sys.executable,'-m','pytest','tests/test_input_adapter.py','tests/test_reinjection_adaptation.py','tests/test_depth_adaptation.py','tests/test_depth_summary.py','-q','--junitxml='+str(xml)],str(reinjection_gate_repo))\n        suites=list(ET.parse(xml).getroot().iter('testsuite'))\n        assert suites and all(int(s.attrib.get('skipped',0))==0 for s in suites)\n        reinjection_gate_state['tests_passed']=sum(int(s.attrib['tests']) for s in suites)\n        parent='/content/r04-final/loop-seed0'\n        corpus='/content/Prophet_AGI/data/fineweb-pilot-v1'\n        common=['--parent-run',parent,'--corpus',corpus]\n        run('initial-equality',[sys.executable,'-u',str(reinjection_gate_repo/'scripts/gate_r04_input_adapter.py'),*common,'--out',str(reinjection_gate_root/'initial-equality.json')],str(reinjection_gate_repo))\n        assert json.loads((reinjection_gate_root/'initial-equality.json').read_text())['passed']\n        for arm in ['fixed_sum','learned_mix']:\n            base=[sys.executable,'-u',str(reinjection_gate_repo/'scripts/adapt_r04_reinjection.py'),*common,'--arm',arm]\n            run(arm+'-preflight',base+['--mode','preflight','--out',str(reinjection_gate_root/(arm+'-preflight'))],str(reinjection_gate_repo))\n            assert json.loads((reinjection_gate_root/(arm+'-preflight')/'preflight.json').read_text())['passed']\n            continuous=reinjection_gate_root/(arm+'-continuous')\n            resumed=reinjection_gate_root/(arm+'-resumed')\n            run(arm+'-continuous8',base+['--out',str(continuous),'--max-session-steps','8'],str(reinjection_gate_repo))\n            run(arm+'-split1',base+['--out',str(resumed),'--max-session-steps','1'],str(reinjection_gate_repo))\n            run(arm+'-split7',base+['--out',str(resumed),'--max-session-steps','7'],str(reinjection_gate_repo))\n            run(arm+'-audit',[sys.executable,'-u',str(reinjection_gate_repo/'scripts/audit_r04_restart.py'),'--left',str(continuous),'--right',str(resumed),'--expected-step','8','--out',str(reinjection_gate_root/(arm+'-restart.json'))],str(reinjection_gate_repo))\n            assert json.loads((reinjection_gate_root/(arm+'-restart.json')).read_text())['passed']\n        histories=[json.loads((reinjection_gate_root/(a+'-continuous')/'evaluation-step-000008.json').read_text())['depth_history'] for a in ['fixed_sum','learned_mix']]\n        assert histories[0]==histories[1]\n        reinjection_gate_state.update(status='all_gates_passed',paired_depth_history=histories[0],continuation_preselected=['fixed_sum-continuous','learned_mix-continuous'])\n        save()\n    except BaseException:\n        reinjection_gate_state['status']='failed'\n        (reinjection_gate_root/'error.txt').write_text(traceback.format_exc())\n        save()\nreinjection_gate_thread=threading.Thread(target=reinjection_gate_queue,daemon=True)\nreinjection_gate_thread.start()\nprint('REINJECTION_GATE_STARTED',reinjection_gate_revision,str(reinjection_gate_root),flush=True)\n")
import zipfile, gc
assert reinjection_gate_thread.is_alive() or reinjection_gate_state['status']=='all_gates_passed'
reinjection_long_root=Path('/content/r04-reinjection-long-v1')
assert not reinjection_long_root.exists()
reinjection_long_root.mkdir()
reinjection_long_processes={}
reinjection_long_state={'status':'waiting_for_gates','revision':reinjection_gate_revision,'maximum_seconds':5400,'outputs':{a:str(reinjection_gate_root/(a+'-continuous')) for a in ['fixed_sum','learned_mix']},'scope':'Continue only the preselected continuous-eight prefixes to 512 under the frozen protocol; no intermediate quality selection.'}
def reinjection_export(root, destination, groups, processes):
    assert not destination.exists()
    entries={}
    for prefix,folder in groups:
        for path in sorted(folder.rglob('*')):
            if path.is_file() and path.suffix in ['.json','.jsonl','.txt','.xml','.py'] and path.name!='export-manifest.json':
                name=(Path(prefix)/path.relative_to(folder)).as_posix()
                entries[name]=path
    manifest={'revision':reinjection_gate_revision,'processes':{k:{'pid':p.pid,'returncode':p.poll()} for k,p in processes.items()},'files':{n:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for n,p in entries.items()}}
    assert all(p['returncode']==0 for p in manifest['processes'].values())
    raw=json.dumps(manifest,indent=2)+chr(10)
    (root/'export-manifest.json').write_text(raw)
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,path in entries.items(): archive.write(path,name)
        archive.writestr('export-manifest.json',raw)
    return {'path':str(destination),'bytes':destination.stat().st_size,'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'files':len(entries)}
def reinjection_long_queue():
    began=None
    def save():
        if began is not None: reinjection_long_state['elapsed_seconds']=time.monotonic()-began
        (reinjection_long_root/'queue.json').write_text(json.dumps(reinjection_long_state,indent=2)+chr(10))
    def run(label,command):
        remaining=5400-(time.monotonic()-began)
        assert remaining>0,'paired deadline exceeded'
        reinjection_long_state.update(status=label,command=command)
        with (reinjection_long_root/(label+'.txt')).open('w') as log:
            p=subprocess.Popen(command,cwd=reinjection_gate_repo,env=reinjection_gate_environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            reinjection_long_processes[label]=p
            reinjection_long_state['pid']=p.pid
            save()
            try: rc=p.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGTERM)
                try: p.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid,signal.SIGKILL)
                    p.wait()
                raise RuntimeError('paired deadline exceeded; process terminated')
        reinjection_long_state.setdefault('processes',{})[label]={'pid':p.pid,'returncode':rc}
        save()
        assert rc==0,label+' failed'
    try:
        save()
        reinjection_gate_thread.join(timeout=1800)
        assert not reinjection_gate_thread.is_alive(),'gate wait expired; no continuation'
        assert reinjection_gate_state['status']=='all_gates_passed','gate failure; no continuation'
        assert all(p.poll()==0 for p in reinjection_gate_processes.values())
        assert reinjection_gate_state['elapsed_seconds']<1800
        for arm in ['fixed_sum','learned_mix']:
            assert json.loads((reinjection_gate_root/(arm+'-restart.json')).read_text())['passed']
            assert json.loads((reinjection_gate_root/(arm+'-preflight')/'preflight.json').read_text())['passed']
        gate_export=reinjection_export(reinjection_gate_root,Path('/content/prophet-r04-reinjection-gates.zip'),[('',reinjection_gate_root)],reinjection_gate_processes)
        reinjection_long_state['gate_export']=gate_export
        assert shutil.disk_usage('/content').free>8_000_000_000
        began=time.monotonic()
        for arm,folder in reinjection_long_state['outputs'].items():
            path=Path(folder)
            before=json.loads((path/'evaluation-step-000008.json').read_text())
            assert before['step']==8 and before['identity']['code']['revision']==reinjection_gate_revision
            command=[sys.executable,'-u',str(reinjection_gate_repo/'scripts/adapt_r04_reinjection.py'),'--parent-run','/content/r04-final/loop-seed0','--corpus','/content/Prophet_AGI/data/fineweb-pilot-v1','--arm',arm,'--out',folder,'--max-session-steps','504','--session-minutes','45']
            run(arm+'-to512',command)
            report=json.loads((path/'evaluation-step-000512.json').read_text())
            assert report['complete'] and report['step']==512 and report['tokens_seen']==8388608 and report['skipped_nonfinite']==0
            reinjection_long_state.setdefault('results',{})[arm]={'step':512,'report_sha256':hashlib.sha256((path/'evaluation-step-000512.json').read_bytes()).hexdigest()}
        audit_code="import json,sys,gc; from pathlib import Path; from scripts.audit_r04_restart import load_run,compare_states; result={}; "+chr(10)+"for arm in ['fixed_sum','learned_mix']:"+chr(10)+" state,report=load_run(Path(sys.argv[1])/(arm+'-continuous'),512); result[arm]={'checkpoint':report['checkpoint'],'state':compare_states(state,state),'identity':report['identity']}; del state; gc.collect()"+chr(10)+"Path(sys.argv[2]).write_text(json.dumps({'passed':True,'arms':result,'scope':'Publication hashes, counters, contracts and finiteness of every checkpoint tensor; not equality between different arms.'},indent=2)+chr(10))"
        run('final-state-audit',[sys.executable,'-u','-c',audit_code,str(reinjection_gate_root),str(reinjection_long_root/'final-state-audit.json')])
        run('screen',[sys.executable,'-u',str(reinjection_gate_repo/'scripts/summarize_reinjection.py'),'--fixed-sum',str(Path(reinjection_long_state['outputs']['fixed_sum'])/'evaluation-step-000512.json'),'--learned-mix',str(Path(reinjection_long_state['outputs']['learned_mix'])/'evaluation-step-000512.json'),'--out',str(reinjection_long_root/'summary.json')])
        summary=json.loads((reinjection_long_root/'summary.json').read_text())
        reinjection_long_state.update(status='paired_training_and_screen_complete',seed0_screen_passed=summary['seed0_screen_passed'],checks=summary['checks'])
        save()
        exported=reinjection_export(reinjection_long_root,Path('/content/prophet-r04-reinjection-final.zip'),[('',reinjection_long_root),*[(a,Path(p)) for a,p in reinjection_long_state['outputs'].items()]],reinjection_long_processes)
        reinjection_long_state['export']=exported
        save()
    except BaseException:
        reinjection_long_state['status']='failed'
        (reinjection_long_root/'error.txt').write_text(traceback.format_exc())
        save()
reinjection_long_thread=threading.Thread(target=reinjection_long_queue,daemon=True)
reinjection_long_thread.start()
print('REINJECTION_CONTINUATION_QUEUED',str(reinjection_long_root),flush=True)
