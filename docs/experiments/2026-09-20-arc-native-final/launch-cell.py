import os, sys, json, time, signal, subprocess, threading, traceback, shutil, hashlib, zipfile, xml.etree.ElementTree as ET
from pathlib import Path
assert 'native_arc_thread' not in globals(), 'Preserve the existing native evaluation queue'
assert reinjection_long_thread.is_alive() or reinjection_long_state['status']=='paired_training_and_screen_complete'
native_arc_revision='46321013c2b3a52eabd296ac173c255cd1369bec'
native_arc_repo=Path('/content/native-arc-code-4632101')
native_arc_root=Path('/content/native-arc-evaluation-v1')
assert not native_arc_repo.exists() and not native_arc_root.exists()
assert shutil.disk_usage('/content').free > 3_000_000_000
native_arc_root.mkdir()
native_arc_source=get_ipython().history_manager.input_hist_raw[-1]
assert 'def native_arc_queue()' in native_arc_source
(native_arc_root/'launch-cell.py').write_text(native_arc_source)
native_arc_state={'status':'waiting_for_paired_training','revision':native_arc_revision,'maximum_execution_seconds':1800,'maximum_wait_seconds':1200,'results':{},'scope':'All six preselected native ARC-Easy runs; inference only; no outcome-based selection or training.'}
native_arc_processes={}
native_arc_environment={**os.environ,'PYTHONPATH':str(native_arc_repo),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','TRITON_F32_DEFAULT':'tf32x3','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
def native_arc_queue():
    began=None
    def save():
        native_arc_state['elapsed_execution_seconds']=None if began is None else time.monotonic()-began
        (native_arc_root/'queue.json').write_text(json.dumps(native_arc_state,indent=2)+chr(10))
    def run(label,command,cwd):
        remaining=1800-(time.monotonic()-began)
        assert remaining>0, 'native evaluation deadline exceeded'
        native_arc_state.update(status=label,command=[str(x) for x in command])
        with (native_arc_root/(label+'.txt')).open('x') as output:
            p=subprocess.Popen([str(x) for x in command],cwd=cwd,env=native_arc_environment,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
            native_arc_processes[label]=p
            native_arc_state['pid']=p.pid
            save()
            try: rc=p.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGTERM)
                try: p.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid,signal.SIGKILL);p.wait()
                raise RuntimeError('native evaluation deadline exceeded; process terminated')
        native_arc_state.setdefault('processes',{})[label]={'pid':p.pid,'returncode':rc}
        save()
        assert rc==0,label+' failed; preserve its log'
    try:
        save()
        waiting=time.monotonic()
        while reinjection_long_thread.is_alive():
            assert time.monotonic()-waiting<1200,'preceding training did not finish in the bounded wait'
            reinjection_long_thread.join(timeout=20)
        assert reinjection_long_state['status']=='paired_training_and_screen_complete'
        assert all(p.poll()==0 for p in reinjection_long_processes.values())
        began=time.monotonic()
        (native_arc_root/'preceding-training-queue.json').write_bytes((reinjection_long_root/'queue.json').read_bytes())
        native_arc_state['preceding_export']=reinjection_long_state['export']
        assert hashlib.sha256(Path(native_arc_state['preceding_export']['path']).read_bytes()).hexdigest()==native_arc_state['preceding_export']['sha256']
        checkout=Path('/content/prophet-recovery/repo')
        run('fetch',['git','fetch','origin','claude/prophet-v03-memory-context'],checkout)
        run('worktree',['git','worktree','add','--detach',native_arc_repo,native_arc_revision],checkout)
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=native_arc_repo,text=True).strip()==native_arc_revision
        xml=native_arc_root/'tests.xml'
        run('tests',[sys.executable,'-m','pytest','tests/test_native_choice_eval.py','tests/test_native_choice_summary.py','tests/test_choice_eval.py','-q','--junitxml='+str(xml)],native_arc_repo)
        suites=list(ET.parse(xml).getroot().iter('testsuite'))
        assert sum(int(s.attrib['tests']) for s in suites)==28
        assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ('errors','failures','skipped'))
        native_arc_state['tests_passed']=28
        tokenizer=Path('/content/Prophet_AGI/data/fineweb-pilot-v1/tokenizer.json')
        run('prepare',[sys.executable,native_arc_repo/'scripts/prepare_arc_native_eval.py','--items','/content/prophet-recovery/arc-evaluation-v1/items','--tokenizer',tokenizer,'--out',native_arc_root/'items'],native_arc_repo)
        expected=json.loads((native_arc_repo/'docs/experiments/2026-09-20-arc-native-protocol/manifest.json').read_bytes())
        assert json.loads((native_arc_root/'items/manifest.json').read_bytes())==expected
        assert hashlib.sha256((native_arc_root/'items/items.jsonl').read_bytes()).hexdigest()==expected['items_sha256']
        native_arc_state['input_manifest']=expected
        models=[('original4096',Path('/content/r04-final/loop-seed0'),4096),('fixed_sum',Path('/content/r04-reinjection-gate-v1/fixed_sum-continuous'),512),('learned_mix',Path('/content/r04-reinjection-gate-v1/learned_mix-continuous'),512)]
        for arm,folder,step in models:
            for depth in (4,6):
                key=arm+'-k'+str(depth)
                report_path=native_arc_root/(key+'.json')
                run(key,[sys.executable,'-u',native_arc_repo/'scripts/eval_arc_native.py','--run',folder,'--step',str(step),'--loop-k',str(depth),'--items',native_arc_root/'items','--tokenizer',tokenizer,'--out',report_path],native_arc_repo)
                report=json.loads(report_path.read_bytes())
                assert report['complete'] and report['arm']==arm and report['loop_k']==depth and report['source']['revision']==native_arc_revision
                assert report['evaluation']['rows']==2376 and report['manifest']==expected
                assert report['numerical_policy']['matmul_allow_tf32'] is False and report['numerical_policy']['cudnn_allow_tf32'] is False
                assert 'A100' in report['runtime']['device']
                if arm!='original4096':
                    published=json.loads((folder/'evaluation-step-000512.json').read_bytes())
                    assert report['checkpoint']['checkpoint']==published['checkpoint']
                    assert report['checkpoint']['training_identity']==published['identity']
                native_arc_state['results'][key]={'report_sha256':hashlib.sha256(report_path.read_bytes()).hexdigest(),'checkpoint':report['checkpoint']['checkpoint'],'seconds':report['seconds']}
                save()
        run('analysis',[sys.executable,native_arc_repo/'scripts/summarize_arc_native.py','--reports',native_arc_root,'--items',native_arc_root/'items','--out',native_arc_root/'summary.json'],native_arc_repo)
        native_arc_state['status']='all_six_and_analysis_complete'
    except BaseException:
        native_arc_state['status']='failed'
        (native_arc_root/'error.txt').write_text(traceback.format_exc())
    finally:
        save()
        export=Path('/content/prophet-native-arc-final.zip')
        assert not export.exists()
        selected=[p for p in sorted(native_arc_root.rglob('*')) if p.is_file() and p.name!='items.jsonl']
        manifest={'files':{str(p.relative_to(native_arc_root)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in selected}}
        with zipfile.ZipFile(export,'x',compression=zipfile.ZIP_DEFLATED) as archive:
            for p in selected: archive.write(p,str(p.relative_to(native_arc_root)))
            archive.writestr('export-manifest.json',json.dumps(manifest,indent=2)+chr(10))
        native_arc_state['export']={'path':str(export),'bytes':export.stat().st_size,'sha256':hashlib.sha256(export.read_bytes()).hexdigest(),'files':len(selected)}
        print('NATIVE_ARC_TERMINAL',json.dumps(native_arc_state),flush=True)
native_arc_thread=threading.Thread(target=native_arc_queue,daemon=True)
native_arc_thread.start()
print('NATIVE_ARC_STARTED',native_arc_state,flush=True)