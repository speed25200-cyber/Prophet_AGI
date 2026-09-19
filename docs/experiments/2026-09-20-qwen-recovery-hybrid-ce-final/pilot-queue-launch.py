import math
import traceback

assert 'recovery_pilot_thread' not in globals(), 'Pilot already queued; inspect its handle'
assert diagnostic_head(recovery_gpu_repo) == recovery_run_protocol['revision']
assert diagnostic_head(cache_recovery_repo) == '25d24cc9f4f14e4a19733042a6b4321540378911'
assert recovery_second_process is not None
assert shutil.disk_usage(recovery_root).free > 35 * 1024**3
assert all(not (recovery_run_root / (arm + '-seed0')).exists() for arm in recovery_run_protocol['arms'][1:])
recovery_pilot_state = {'status': 'waiting_for_live_first_segment', 'results': {}, 'protocol': recovery_run_protocol, 'segments': []}
recovery_pilot_processes = {}
recovery_pilot_state_path = recovery_run_root / 'pilot-queue.json'
assert not recovery_pilot_state_path.exists()
recovery_persistence_state = {'status': 'waiting_for_first_final', 'remote_flush_verified': False}
recovery_persistence_thread = None

def pilot_write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)

def pilot_save():
    pilot_write(recovery_pilot_state_path, recovery_pilot_state)

def pilot_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def pilot_report(run):
    paths = sorted(run.glob('evaluation-step-*.json'))
    return json.loads(paths[-1].read_bytes()) if paths else None

def pilot_check_report(report, objective):
    assert 0 < report['step'] <= 2048
    assert report['tokens_seen'] == report['step'] * 2048
    assert report['skipped_nonfinite'] == 0
    assert report['teacher_tokens_seen'] == (report['tokens_seen'] if objective == 'kl' else 0)
    identity = report['identity']
    assert identity['objective'] == objective and identity['precision'] == 'float32'
    assert identity['evaluation_batch_size'] == 1 and identity['evaluation_loop_k'] == 5
    for key in ('train_sha256', 'validation_sha256'):
        assert identity[key] == recovery_run_protocol[key]
    evaluation = report['evaluation']
    assert len(evaluation['documents']) == 372
    assert evaluation['scored_tokens'] == 366762 and evaluation['scored_bytes'] == 1722551
    assert math.isfinite(evaluation['nats_per_token'])

def pilot_run_process(key, command, cwd, environment, log_path, timeout):
    with log_path.open('x') as log:
        process = subprocess.Popen(command, cwd=cwd, env=environment, stdout=log, stderr=subprocess.STDOUT)
        recovery_pilot_processes[key] = process
        recovery_pilot_state['current_process'] = {'key': key, 'pid': process.pid, 'command': command, 'log': str(log_path)}
        pilot_save()
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=180)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=60)
            raise RuntimeError('Bounded process timed out: ' + key)
    assert code == 0, (key, code)

def pilot_persist_first(run, report, audit_path):
    try:
        destination = persistent / 'donor-recovery/fp32-pilot-v1/hybrid-ce-seed0-step002048'
        assert persistent.is_dir() and not destination.exists()
        meta = report['checkpoint']
        assert shutil.disk_usage(persistent).free > meta['bytes'] + 250_000_000
        destination.mkdir(parents=True)
        recovery_persistence_state.update(status='copying_first_final', destination=str(destination), checkpoint=meta, files={})
        state_path = recovery_run_root / 'pilot-first-persistence.json'
        pilot_write(state_path, recovery_persistence_state)
        checkpoint_name = 'ckpt_slot' + str(meta['slot']) + '.pt'
        names = [checkpoint_name, 'recovery.json', 'evaluation-step-002048.json', 'training.jsonl']
        for name in names:
            source, target = run / name, destination / name
            with source.open('rb') as src, target.open('xb') as dst:
                shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
            actual = pilot_hash(target)
            assert actual == pilot_hash(source)
            assert target.stat().st_size == source.stat().st_size
            if name == checkpoint_name:
                assert actual == meta['sha256'] and target.stat().st_size == meta['bytes']
            recovery_persistence_state['files'][name] = {'sha256': actual, 'bytes': target.stat().st_size}
        source_manifest = json.loads((run / 'manifest.json').read_bytes())
        assert meta in source_manifest['checkpoints']
        (destination / 'source_manifest.json').write_text(json.dumps(source_manifest, indent=2))
        (destination / 'manifest.json').write_text(json.dumps({'checkpoints': [meta]}, indent=2))
        (destination / 'plan.json').write_text(json.dumps(recovery_run_protocol, indent=2))
        (destination / 'checkpoint-audit.json').write_bytes(audit_path.read_bytes())
        recovery_persistence_state['status'] = 'mounted_copy_verified_pending_flush_remount'
        pilot_write(state_path, recovery_persistence_state)
        (destination / 'copy-report.json').write_text(json.dumps(recovery_persistence_state, indent=2))
    except Exception:
        recovery_persistence_state.update(status='failed', error=traceback.format_exc())
        pilot_write(recovery_run_root / 'pilot-first-persistence.json', recovery_persistence_state)

def pilot_work():
    global recovery_persistence_thread
    try:
        recovery_second_thread.join(timeout=5400)
        assert not recovery_second_thread.is_alive(), 'Existing worker remains live; no restart'
        assert recovery_second_process.poll() == 0, 'Existing worker failed; inspect before restart'
        assert recovery_second_state['status'] in ('training_complete_pending_audit_and_persistence', 'another_segment_required')
        for arm in recovery_run_protocol['arms']:
            core, objective = arm.split('-')
            run = recovery_run_root / (arm + '-seed0')
            report = pilot_report(run)
            if report is not None:
                pilot_check_report(report, objective)
            current = 0 if report is None else report['step']
            segment = 0
            while current < 2048:
                segment += 1
                assert segment <= 4, 'Repeated incomplete segments require inspection'
                assert diagnostic_head(recovery_gpu_repo) == recovery_run_protocol['revision']
                assert shutil.disk_usage(recovery_root).free > 12 * 1024**3
                command = list(recovery_second_command)
                replacements = {'--initialization': str(recovery_initializations / (core + '.pt')), '--audit': str(recovery_initializations / (core + '-conversion.json')), '--out': str(run), '--objective': objective, '--max-session-steps': str(2048-current), '--session-minutes': '40'}
                for flag, value in replacements.items():
                    command[command.index(flag) + 1] = value
                if objective == 'kl':
                    command += ['--alpha', '0.5', '--temperature', '1.0']
                key = arm + '-queued-segment' + str(segment)
                recovery_pilot_state.update(status='training_' + arm, current_step=current)
                pilot_run_process(key, command, recovery_gpu_repo, recovery_gpu_environment, recovery_run_root / (key + '.txt'), 3600)
                report = pilot_report(run)
                assert report is not None and report['step'] > current
                pilot_check_report(report, objective)
                recovery_pilot_state['segments'].append({'arm': arm, 'start': current, 'end': report['step'], 'log': key + '.txt', 'command': command})
                current = report['step']
                pilot_save()
            assert report is not None and current == 2048
            rows = [json.loads(line) for line in (run / 'training.jsonl').read_text().splitlines()]
            assert len(rows) == 2048
            for expected, row in enumerate(rows, 1):
                assert row['step'] == expected and row['tokens'] == expected * 2048
                assert math.isfinite(row['loss']) and math.isfinite(row['extra']['train/grad_norm'])
            audit_path = recovery_run_root / (arm + '-final-audit.json')
            recovery_pilot_state['status'] = 'auditing_' + arm
            command = [str(recovery_python), str(cache_recovery_repo / 'scripts/audit_recovery_checkpoint.py'), '--run', str(run), '--step', '2048', '--out', str(audit_path)]
            pilot_run_process(arm + '-audit', command, cache_recovery_repo, recovery_setup_environment, recovery_run_root / (arm + '-final-audit.txt'), 1200)
            audit = json.loads(audit_path.read_bytes())
            assert audit['complete'] and audit['step'] == 2048
            assert audit['checkpoint'] == report['checkpoint']
            recovery_pilot_state['results'][arm] = {'step': 2048, 'tokens_seen': report['tokens_seen'], 'checkpoint': report['checkpoint'], 'nats_per_token': report['evaluation']['nats_per_token'], 'bits_per_byte': report['evaluation']['bits_per_byte'], 'audit_sha256': pilot_hash(audit_path), 'evaluation_sha256': pilot_hash(run / 'evaluation-step-002048.json'), 'training_rows': len(rows), 'remote_flush_verified': False}
            pilot_save()
            if arm == 'hybrid-ce':
                recovery_persistence_thread = threading.Thread(target=pilot_persist_first, args=(run, report, audit_path), name='first-final-mounted-copy', daemon=True)
                recovery_persistence_thread.start()
        recovery_pilot_state['status'] = 'four_arms_complete_local_audits_passed_remote_persistence_pending'
    except Exception:
        recovery_pilot_state.update(status='failed', error=traceback.format_exc())
    finally:
        pilot_save()

pilot_save()
recovery_pilot_thread = threading.Thread(target=pilot_work, name='frozen-four-arm-pilot', daemon=True)
recovery_pilot_thread.start()
print('PILOT_QUEUED', recovery_pilot_state, flush=True)
print('FIRST_PROCESS', recovery_second_process.pid, recovery_second_process.poll(), flush=True)