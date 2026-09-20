import os, sys, json, time, signal, subprocess, threading, traceback, shutil, hashlib, xml.etree.ElementTree as ET
from pathlib import Path
assert all(p.poll() == 0 for group in [depth_long_processes, depth_analysis_processes, recurrence_processes] for p in group.values())
reinjection_gate_revision = 'dce35abc439fdeeb8f1482b1219dee3cc242c267'
reinjection_gate_root = Path('/content/r04-reinjection-gate-v1')
reinjection_gate_repo = Path('/content/r04-reinjection-code-dce35ab')
assert not reinjection_gate_root.exists() and not reinjection_gate_repo.exists()
assert shutil.disk_usage('/content').free > 32_000_000_000
reinjection_gate_root.mkdir()
reinjection_gate_processes = {}
reinjection_gate_state = {'status':'starting','revision':reinjection_gate_revision,'maximum_seconds':1800,'scope':'Initial actual-shape equality, CUDA tests, both k6 preflights and both separate-process 8 versus 1+7 restart gates. No long continuation.'}
reinjection_gate_environment = {**os.environ,'PYTHONPATH':str(reinjection_gate_repo),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','TRITON_F32_DEFAULT':'tf32x3','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
def reinjection_gate_queue():
    began = time.monotonic()
    def save():
        reinjection_gate_state['elapsed_seconds'] = time.monotonic()-began
        (reinjection_gate_root/'queue.json').write_text(json.dumps(reinjection_gate_state,indent=2)+chr(10))
    def run(label, command, cwd=None):
        remaining = 1800-(time.monotonic()-began)
        assert remaining > 0, 'gate deadline exceeded'
        reinjection_gate_state.update(status=label,command=command)
        with (reinjection_gate_root/(label+'.txt')).open('w') as output:
            p = subprocess.Popen(command,cwd=cwd,env=reinjection_gate_environment,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
            reinjection_gate_processes[label] = p
            reinjection_gate_state['pid'] = p.pid
            save()
            try:
                rc=p.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGTERM)
                try: p.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid,signal.SIGKILL)
                    p.wait()
                raise RuntimeError('gate deadline exceeded; process terminated')
        reinjection_gate_state.setdefault('processes',{})[label]={'pid':p.pid,'returncode':rc}
        save()
        assert rc == 0, label+' failed; inspect its log'
    try:
        checkout=Path('/content/prophet-recovery/repo')
        assert checkout.exists()
        run('fetch',['git','fetch','origin','claude/prophet-v03-memory-context'],str(checkout))
        run('worktree',['git','worktree','add','--detach',str(reinjection_gate_repo),reinjection_gate_revision],str(checkout))
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=reinjection_gate_repo,text=True).strip()==reinjection_gate_revision
        xml=reinjection_gate_root/'tests.xml'
        run('tests',[sys.executable,'-m','pytest','tests/test_input_adapter.py','tests/test_reinjection_adaptation.py','tests/test_depth_adaptation.py','tests/test_depth_summary.py','-q','--junitxml='+str(xml)],str(reinjection_gate_repo))
        suites=list(ET.parse(xml).getroot().iter('testsuite'))
        assert suites and all(int(s.attrib.get('skipped',0))==0 for s in suites)
        reinjection_gate_state['tests_passed']=sum(int(s.attrib['tests']) for s in suites)
        parent='/content/r04-final/loop-seed0'
        corpus='/content/Prophet_AGI/data/fineweb-pilot-v1'
        common=['--parent-run',parent,'--corpus',corpus]
        run('initial-equality',[sys.executable,'-u',str(reinjection_gate_repo/'scripts/gate_r04_input_adapter.py'),*common,'--out',str(reinjection_gate_root/'initial-equality.json')],str(reinjection_gate_repo))
        assert json.loads((reinjection_gate_root/'initial-equality.json').read_text())['passed']
        for arm in ['fixed_sum','learned_mix']:
            base=[sys.executable,'-u',str(reinjection_gate_repo/'scripts/adapt_r04_reinjection.py'),*common,'--arm',arm]
            run(arm+'-preflight',base+['--mode','preflight','--out',str(reinjection_gate_root/(arm+'-preflight'))],str(reinjection_gate_repo))
            assert json.loads((reinjection_gate_root/(arm+'-preflight')/'preflight.json').read_text())['passed']
            continuous=reinjection_gate_root/(arm+'-continuous')
            resumed=reinjection_gate_root/(arm+'-resumed')
            run(arm+'-continuous8',base+['--out',str(continuous),'--max-session-steps','8'],str(reinjection_gate_repo))
            run(arm+'-split1',base+['--out',str(resumed),'--max-session-steps','1'],str(reinjection_gate_repo))
            run(arm+'-split7',base+['--out',str(resumed),'--max-session-steps','7'],str(reinjection_gate_repo))
            run(arm+'-audit',[sys.executable,'-u',str(reinjection_gate_repo/'scripts/audit_r04_restart.py'),'--left',str(continuous),'--right',str(resumed),'--expected-step','8','--out',str(reinjection_gate_root/(arm+'-restart.json'))],str(reinjection_gate_repo))
            assert json.loads((reinjection_gate_root/(arm+'-restart.json')).read_text())['passed']
        histories=[json.loads((reinjection_gate_root/(a+'-continuous')/'evaluation-step-000008.json').read_text())['depth_history'] for a in ['fixed_sum','learned_mix']]
        assert histories[0]==histories[1]
        reinjection_gate_state.update(status='all_gates_passed',paired_depth_history=histories[0],continuation_preselected=['fixed_sum-continuous','learned_mix-continuous'])
        save()
    except BaseException:
        reinjection_gate_state['status']='failed'
        (reinjection_gate_root/'error.txt').write_text(traceback.format_exc())
        save()
reinjection_gate_thread=threading.Thread(target=reinjection_gate_queue,daemon=True)
reinjection_gate_thread.start()
print('REINJECTION_GATE_STARTED',reinjection_gate_revision,str(reinjection_gate_root),flush=True)
