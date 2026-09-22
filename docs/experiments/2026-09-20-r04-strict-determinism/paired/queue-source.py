def hash_probe_run():
    started=time.monotonic()
    try:
        for label in ['strict-a','strict-b']:
            command=[sys.executable,'-u',str(hash_probe_root/'probe.py'),'--model-repo',str(adaptation_repo),'--policy','strict','--trace',str(hash_probe_root/(label+'.json')),'--parent-run',str(final_stage_root/'loop-seed0'),'--corpus',str(corpus),'--arm','fixed4','--mode','preflight','--out',str(hash_probe_root/label)]
            hash_probe_state.update(status=label,command=command)
            with (hash_probe_root/(label+'.txt')).open('x') as log:
                p=subprocess.Popen(command,cwd=adaptation_repo,env=adaptation_environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                hash_probe_processes[label]=p
                hash_probe_state['pid']=p.pid
                try: code=p.wait(timeout=600)
                except subprocess.TimeoutExpired:
                    p.terminate()
                    try: p.wait(timeout=20)
                    except subprocess.TimeoutExpired: p.kill(); p.wait()
                    raise
                assert code==0,(label,code)
        a,b=[json.loads((hash_probe_root/(label+'.json')).read_text()) for label in ['strict-a','strict-b']]
        assert a['complete'] and b['complete']
        comparison={'initial_equal':a['initial_weights']==b['initial_weights'],'steps':[],'final_different':[k for k in a['final_weights'] if a['final_weights'][k]!=b['final_weights'][k]]}
        for x,y in zip(a['steps'],b['steps']):
            comparison['steps'].append({'inputs_equal':x['inputs']==y['inputs'],'rng_before_equal':x['rng_before_forward']==y['rng_before_forward'],'rng_after_equal':x['rng_after_backward']==y['rng_after_backward'],'different_gradients':[k for k in x['gradients'] if x['gradients'][k]!=y['gradients'][k]]})
        (hash_probe_root/'comparison.json').write_text(json.dumps(comparison,indent=2)+chr(10))
        hash_probe_state.update(status='complete',comparison=comparison)
    except BaseException as exc:
        hash_probe_state.update(status='stopped_requires_inspection',error=repr(exc))
        (hash_probe_root/'error.txt').write_text(traceback.format_exc())
    hash_probe_state['elapsed_seconds']=time.monotonic()-started
    (hash_probe_root/'queue.json').write_text(json.dumps(hash_probe_state,indent=2)+chr(10))
