"""Matched report analysis with analytically known contrasts and corrupt evidence."""
import copy
import hashlib
import json
import math

import pytest

from scripts.recover_qwen import recovery_config
from scripts.summarize_recovery_pilot import ARMS, summarize
from tests.test_training import tiny_model_config


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def evaluation(ce, depth=2):
    docs = [{'index': i, 'sha256': str(i), 'total_nats': ce * n,
             'scored_tokens': n, 'scored_bytes': n * 2} for i, n in enumerate((10, 20))]
    return {'documents': docs, 'total_nats': ce * 30, 'scored_tokens': 30, 'scored_bytes': 60,
            'nats_per_token': ce, 'bits_per_byte': ce / 2 / math.log(2), 'seq_len': 8,
            'batch_size': 1, 'loop_k': depth, 'precision': 'fp32', 'protocol': 'test windows'}


@pytest.fixture
def pilot(tmp_path):
    root, baselines = tmp_path / 'runs', tmp_path / 'baselines'
    plan = {'protocol': 'qwen-recovery-fp32-pilot-v1', 'arms': list(ARMS), 'seed': 0,
            'steps_per_arm': 2, 'batch_size': 1, 'seq_len': 8, 'grad_accum': 1,
            'input_tokens_per_arm': 16, 'precision': 'float32', 'allow_tf32': False,
            'loop_k': 2, 'evaluation_batch_size': 1, 'muon_peak_lr': 0.01, 'adamw_peak_lr': 0.001,
            'train_sha256': 'train', 'validation_sha256': 'validation', 'kl_alpha': 0.5, 'kl_temperature': 1.0}
    save(root / 'plan.json', plan)
    identity = {'donor_revision': 'source', 'donor_weights_sha256': 'source-weights', 'donor_config': {},
                'tokenizer': {'fixture': True}, 'train_sha256': 'train', 'validation_sha256': 'validation',
                'torch': 'fixture', 'transformers': 'fixture', 'cuda': None, 'fla': None,
                'triton_f32_default': 'tf32x3', 'precision': 'float32', 'teacher_dtype': 'float32',
                'evaluation_loop_k': 2, 'evaluation_batch_size': 1}
    runtime = {k: identity[k] for k in ('torch', 'transformers', 'cuda', 'fla', 'triton_f32_default')}
    runtime.update(device='cpu', requested_precision='float32', allow_tf32_matmul=False, allow_tf32_cudnn=False)
    configs = {}
    for core, ce in [('donor', 1), ('hybrid', 10), ('attention', 8)]:
        config = tiny_model_config().to_dict()
        config['recurrent']['core_pattern'] = ['full_attn'] if core == 'attention' else ['gdn']
        configs[core] = config
        save(baselines / f'baseline-{core}.json', {
            **identity, 'config': config, 'complete': True, 'trained_steps': 0,
            'initialization_sha256': core, 'runtime': runtime, 'unique_parameters': 100,
            'evaluation': evaluation(ce, None if core == 'donor' else 2)})
    for arm, ce in zip(ARMS, (4, 3.5, 3, 3.2), strict=True):
        core, objective = arm.split('-')
        own_identity = {**identity, 'objective': objective, 'initialization_sha256': core}
        contract = {'run_identity': own_identity, 'total_steps': 2, 'batch_size': 1, 'seq_len': 8,
                    'grad_accum_steps': 1, 'peak_lr_muon': 0.01, 'peak_lr_adamw': 0.001,
                    'seed': 0, 'dtype': 'float32', 'allow_tf32': False, 'device_type': 'cpu',
                    'loss_chunk_tokens': 3, 'schedule': {'warmup': 100}}
        if objective == 'kl':
            contract['distillation'] = {'format': 'frozen-forward-kl-v1', 'identity': own_identity,
                'state_sha256': 'a' * 64, 'alpha': 0.5, 'temperature': 1.0, 'chunk_tokens': 3,
                'alignment': 'next-token; exclude final position; full vocabulary'}
        checkpoint = {'step': 2, 'slot': 0, 'sha256': arm, 'bytes': 1, 'extra': {}}
        run = root / (arm + '-seed0')
        config = recovery_config(configs[core], 2, 8).to_dict()
        report = {'step': 2, 'tokens_seen': 16, 'teacher_tokens_seen': 16 if objective == 'kl' else 0,
                  'identity': own_identity, 'checkpoint': checkpoint, 'skipped_nonfinite': 0,
                  'evaluation': evaluation(ce)}
        save(run / 'evaluation-step-000002.json', report)
        save(run / 'recovery.json', {'training_contract': contract, 'config': config})
        save(run / 'manifest.json', {'checkpoints': [checkpoint]})
        save(root / (arm + '-final-audit.json'), {
            'protocol': 'evaluated-recovery-checkpoint-audit-v1', 'complete': True,
            'evaluation_sha256': hashlib.sha256((run / 'evaluation-step-000002.json').read_bytes()).hexdigest(),
            'step': 2, 'tokens_seen': 16, 'checkpoint': checkpoint, 'skipped_nonfinite': 0,
            'finite_tensor_counts': {'model': 1, 'optimizers': 1}, 'training_contract': contract,
            'config': config, 'evaluation_nats_per_token': ce})
        (run / 'training.jsonl').write_text('\n'.join(json.dumps({
            'step': step, 'tokens': step * 8, 'loss': ce, 'lr': 0.001, 'seconds': step,
            'extra': {'train/grad_norm': 1.0}}) for step in (1, 2)), encoding='utf-8')
    return root, baselines


def test_four_arm_contrasts_have_known_direction_intervals_and_cost_scope(pilot):
    result = summarize(*pilot, draws=100)
    assert result['complete'] and result['ce_ranking'] == ['hybrid-kl', 'attention-kl', 'attention-ce', 'hybrid-ce']
    for name, expected in [('hybrid_minus_attention_ce', 0.5), ('hybrid_minus_attention_kl', -0.2),
                           ('kl_minus_ce_hybrid', -1.0), ('kl_minus_ce_attention', -0.3)]:
        assert result['contrasts'][name]['ce_delta'] == pytest.approx(expected)
        assert result['contrasts'][name]['ce_document_bootstrap_95'] == pytest.approx([expected, expected])
    assert result['arms']['hybrid-ce']['fraction_initial_ce_gap_closed'] == pytest.approx(2/3)
    assert result['arms']['hybrid-ce']['teacher_tokens'] == 0
    assert result['arms']['hybrid-kl']['teacher_tokens'] == 16
    assert result['arms']['hybrid-ce']['median_logged_update_seconds_last_256'] == 1.5
    assert 'no architecture adoption' in result['scope'] and 'unadjusted' in result['uncertainty']


@pytest.mark.parametrize('corruption', ['missing_arm', 'report_hash', 'rotated', 'different_lr',
                                      'different_batch', 'different_initialization', 'different_teacher',
                                      'teacher_tokens', 'short_log', 'nonfinite_baseline'])
def test_four_arm_summary_rejects_unmatched_or_incomplete_evidence(pilot, corruption):
    root, baselines = pilot
    arm = 'attention-kl'
    run = root / (arm + '-seed0')
    report_path = run / 'evaluation-step-000002.json'
    audit_path = root / (arm + '-final-audit.json')
    report, audit = json.loads(report_path.read_bytes()), json.loads(audit_path.read_bytes())
    if corruption == 'missing_arm':
        report_path.unlink()
    elif corruption == 'rotated':
        save(run / 'manifest.json', {'checkpoints': []})
    elif corruption == 'short_log':
        path = run / 'training.jsonl'
        path.write_text(path.read_text().splitlines()[0])
    elif corruption == 'nonfinite_baseline':
        path = baselines / 'baseline-donor.json'
        baseline = json.loads(path.read_bytes())
        baseline['evaluation']['documents'][0]['total_nats'] = float('nan')
        save(path, baseline)
    else:
        if corruption == 'report_hash':
            report['evaluation']['nats_per_token'] += 0.1
        elif corruption == 'different_lr':
            audit['training_contract']['peak_lr_muon'] *= 2
        elif corruption == 'different_batch':
            report['evaluation']['batch_size'] = 2
        elif corruption == 'different_initialization':
            report['identity']['initialization_sha256'] = 'other'
            audit['training_contract']['run_identity'] = copy.deepcopy(report['identity'])
        elif corruption == 'different_teacher':
            audit['training_contract']['distillation']['state_sha256'] = 'b' * 64
        elif corruption == 'teacher_tokens':
            report['teacher_tokens_seen'] = 0
        save(report_path, report)
        if corruption != 'report_hash':
            audit['evaluation_sha256'] = hashlib.sha256(report_path.read_bytes()).hexdigest()
            save(audit_path, audit)
            save(run / 'recovery.json', {'config': audit['config'], 'training_contract': audit['training_contract']})
    expected_error = {'missing_arm': 'evaluation-step', 'report_hash': 'evaluation hash',
                      'rotated': 'rotated', 'different_lr': 'planned training setting',
                      'different_batch': 'execution setting', 'different_initialization': 'initialization identity',
                      'different_teacher': 'KL teacher', 'teacher_tokens': 'teacher token',
                      'short_log': 'training row count', 'nonfinite_baseline': 'invalid document metrics'}
    with pytest.raises((ValueError, FileNotFoundError), match=expected_error[corruption]):
        summarize(root, baselines, draws=100)
