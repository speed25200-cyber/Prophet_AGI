#!/usr/bin/env python3
"""Compare all four audited recovery endpoints; no automatic architecture adoption.

Reads report evidence rather than reloading weights. Checkpoint integrity and
finite-state claims depend on the separately executed checkpoint auditor.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prophet.eval.paired import paired_document_difference  # noqa: E402
from scripts.audit_recovery_checkpoint import require  # noqa: E402
from scripts.recover_qwen import digest, recovery_config, write_report  # noqa: E402

ARMS = ('hybrid-ce', 'attention-ce', 'hybrid-kl', 'attention-kl')
CONTRASTS = {'hybrid_minus_attention_ce': ('hybrid-ce', 'attention-ce'),
             'hybrid_minus_attention_kl': ('hybrid-kl', 'attention-kl'),
             'kl_minus_ce_hybrid': ('hybrid-kl', 'hybrid-ce'),
             'kl_minus_ce_attention': ('attention-kl', 'attention-ce')}


def read(path):
    return json.loads(path.read_bytes())


def verify_evaluation(evaluation):
    require(bool(evaluation['documents']), 'empty evaluation')
    require(all(math.isfinite(doc['total_nats']) and doc['total_nats'] >= 0 and
                doc['scored_tokens'] > 0 and doc['scored_bytes'] > 0 for doc in evaluation['documents']),
            'invalid document metrics')
    for key in ('total_nats', 'scored_tokens', 'scored_bytes'):
        total = sum(doc[key] for doc in evaluation['documents'])
        require(math.isfinite(total) and total > 0 and math.isclose(
            total, evaluation[key], rel_tol=1e-10), 'evaluation aggregate differs')
    require(math.isclose(evaluation['nats_per_token'], evaluation['total_nats'] /
                         evaluation['scored_tokens'], rel_tol=1e-10), 'evaluation CE differs')
    require(math.isclose(evaluation['bits_per_byte'], evaluation['total_nats'] /
                         evaluation['scored_bytes'] / math.log(2), rel_tol=1e-10), 'evaluation BPB differs')


def summarize(root, baselines, *, draws=10000):
    plan = read(root / 'plan.json')
    require(plan['protocol'] == 'qwen-recovery-fp32-pilot-v1', 'unknown pilot protocol')
    require(plan['arms'] == list(ARMS), 'four matched arms are required')
    step, seed = plan['steps_per_arm'], plan['seed']
    tokens = step * plan['batch_size'] * plan['seq_len'] * plan['grad_accum']
    require(step > 0 and tokens == plan['input_tokens_per_arm'], 'planned token budget differs')
    require(plan['precision'] == 'float32' and plan['allow_tf32'] is False, 'pilot precision differs')
    references = {name: read(baselines / f'baseline-{name}.json') for name in ('donor', 'hybrid', 'attention')}
    for baseline in references.values():
        require(baseline['complete'] and baseline['trained_steps'] == 0, 'baseline is not an initialization')
        verify_evaluation(baseline['evaluation'])
    reports, arms, identities, contracts, distillation = {}, {}, {}, {}, {}
    for arm in ARMS:
        core, objective = arm.split('-')
        run = root / f'{arm}-seed{seed}'
        report_path = run / f'evaluation-step-{step:06d}.json'
        report = read(report_path)
        audit_path = root / f'{arm}-final-audit.json'
        audit = read(audit_path)
        manifest = read(run / 'recovery.json')
        contract, identity = audit['training_contract'], report['identity']
        require(audit['protocol'] == 'evaluated-recovery-checkpoint-audit-v1' and audit['complete'],
                'incomplete or incompatible checkpoint audit')
        require(audit['evaluation_sha256'] == digest(report_path), 'evaluation hash differs from audit')
        require(report['step'] == audit['step'] == step and report['tokens_seen'] == audit['tokens_seen'] == tokens,
                'endpoint step or token budget differs')
        require(report['checkpoint'] == audit['checkpoint'] and report['checkpoint']['step'] == step,
                'checkpoint identity differs')
        require(report['checkpoint'] in read(run / 'manifest.json')['checkpoints'], 'evaluated slot has rotated')
        require(report['skipped_nonfinite'] == audit['skipped_nonfinite'] == 0, 'nonfinite updates were skipped')
        require(set(audit['finite_tensor_counts']) == {'model', 'optimizers'} and
                all(v > 0 for v in audit['finite_tensor_counts'].values()), 'finite-state evidence is missing')
        require(contract == manifest['training_contract'] and contract['run_identity'] == identity,
                'training contract or evaluation identity differs')
        if contract['device_type'] == 'cuda':
            require(audit['cuda_rng_saved'], 'CUDA RNG evidence is missing')
        expected_config = json.loads(json.dumps(recovery_config(
            references[core]['config'], plan['loop_k'], plan['seq_len']).to_dict()))
        require(audit['config'] == manifest['config'] == expected_config, 'recovered model configuration differs')
        require(identity['initialization_sha256'] == references[core]['initialization_sha256'],
                'initialization identity differs')
        require(identity['objective'] == objective, 'objective label differs')
        require(report['teacher_tokens_seen'] == (tokens if objective == 'kl' else 0), 'teacher token count differs')
        for key in ('donor_revision', 'donor_weights_sha256', 'donor_config', 'tokenizer', 'validation_sha256'):
            require(all(identity[key] == baseline[key] for baseline in references.values()),
                    'baseline input identity differs: ' + key)
        require(identity['train_sha256'] == plan['train_sha256'] and
                identity['validation_sha256'] == plan['validation_sha256'], 'planned corpus differs')
        for key, expected in {'total_steps': step, 'batch_size': plan['batch_size'],
                              'seq_len': plan['seq_len'], 'grad_accum_steps': plan['grad_accum'],
                              'peak_lr_muon': plan['muon_peak_lr'], 'peak_lr_adamw': plan['adamw_peak_lr'],
                              'seed': seed, 'dtype': plan['precision'], 'allow_tf32': False}.items():
            require(contract[key] == expected, 'planned training setting differs: ' + key)
        require(identity['precision'] == identity['teacher_dtype'] == 'float32' and identity['evaluation_loop_k'] == plan['loop_k'] and
                identity['evaluation_batch_size'] == plan['evaluation_batch_size'], 'evaluation policy differs')
        evaluation = report['evaluation']
        verify_evaluation(evaluation)
        require(evaluation['batch_size'] == plan['evaluation_batch_size'] and
                evaluation['seq_len'] == plan['seq_len'] and evaluation['loop_k'] == plan['loop_k'] and
                evaluation['precision'] == 'fp32', 'evaluation execution setting differs')
        require(audit['evaluation_nats_per_token'] == evaluation['nats_per_token'], 'audited loss differs')
        for baseline in references.values():
            require(all(evaluation[key] == baseline['evaluation'][key] for key in
                        ('batch_size', 'seq_len', 'precision', 'protocol')), 'baseline evaluation protocol differs')
            for key in ('torch', 'transformers', 'cuda', 'fla', 'triton_f32_default'):
                require(identity[key] == baseline['runtime'][key], 'baseline runtime differs: ' + key)
            require(contract['device_type'] == baseline['runtime']['device'], 'baseline device differs')
            require(baseline['runtime']['requested_precision'] == plan['precision'] and
                    baseline['runtime']['allow_tf32_matmul'] is False and
                    baseline['runtime']['allow_tf32_cudnn'] is False, 'baseline precision policy differs')
            require(len(evaluation['documents']) == len(baseline['evaluation']['documents']), 'baseline document count differs')
            for left, right in zip(evaluation['documents'], baseline['evaluation']['documents'], strict=True):
                require(all(left[key] == right[key] for key in ('index', 'sha256', 'scored_tokens', 'scored_bytes')),
                        'baseline document identity differs')
        if objective == 'kl':
            teacher = copy.deepcopy(contract['distillation'])
            require(teacher.pop('identity') == identity, 'teacher input identity differs')
            require(teacher['format'] == 'frozen-forward-kl-v1' and
                    teacher['alpha'] == plan['kl_alpha'] and teacher['temperature'] == plan['kl_temperature'],
                    'distillation settings differ')
            require(teacher['alignment'] == 'next-token; exclude final position; full vocabulary' and
                    teacher['chunk_tokens'] == contract['loss_chunk_tokens'], 'distillation alignment differs')
            distillation[core] = teacher
        else:
            require('distillation' not in contract, 'CE arm contains distillation')
        identities[arm] = {k: v for k, v in identity.items() if k not in ('initialization_sha256', 'objective')}
        contracts[arm] = {k: v for k, v in contract.items() if k not in ('run_identity', 'distillation')}
        logs = [json.loads(line) for line in (run / 'training.jsonl').read_text().splitlines()]
        require(len(logs) == step, 'training row count differs')
        for expected, row in enumerate(logs, 1):
            require(row['step'] == expected and row['tokens'] == expected * tokens // step, 'training row budget differs')
            require(all(math.isfinite(row[key]) for key in ('loss', 'lr', 'seconds')) and row['seconds'] > 0,
                    'invalid training metrics')
            require(math.isfinite(row['extra']['train/grad_norm']), 'invalid gradient norm')
        before = references[core]['evaluation']['nats_per_token']
        donor = references['donor']['evaluation']['nats_per_token']
        require(before > donor, 'gap closure requires a worse initialization than donor')
        arms[arm] = {'nats_per_token': evaluation['nats_per_token'], 'bits_per_byte': evaluation['bits_per_byte'],
                     'unique_parameters': references[core]['unique_parameters'], 'checkpoint': report['checkpoint'],
                     'evaluation_sha256': digest(report_path), 'audit_sha256': digest(audit_path),
                     'student_tokens': tokens, 'teacher_tokens': report['teacher_tokens_seen'],
                     'initial_ce': before, 'fraction_initial_ce_gap_closed': (before-evaluation['nats_per_token'])/(before-donor),
                     'median_logged_update_seconds_last_256': statistics.median(row['seconds'] for row in logs[-256:]),
                     'timing_samples': min(256, len(logs)),
                     'versus_own_initialization': paired_document_difference(evaluation, references[core]['evaluation'], draws=draws)}
        reports[arm] = evaluation
    require(all(identity == identities[ARMS[0]] for identity in identities.values()), 'arm runtime/input identities differ')
    require(all(contract == contracts[ARMS[0]] for contract in contracts.values()), 'arm schedules/settings differ')
    require(distillation['hybrid'] == distillation['attention'], 'KL teacher tensor identity/settings differ')
    return {'protocol': 'four-arm-recovery-comparison-v1', 'complete': True, 'plan': plan, 'arms': arms,
            'contrasts': {name: paired_document_difference(reports[left], reports[right], draws=draws)
                          for name, (left, right) in CONTRASTS.items()},
            'ce_ranking': sorted(ARMS, key=lambda name: arms[name]['nats_per_token']),
            'donor_nats_per_token': references['donor']['evaluation']['nats_per_token'],
            'scope': 'single-seed, matched student tokens; KL adds teacher compute; no architecture adoption',
            'uncertainty': 'four factorial contrasts; all document intervals unadjusted and conditional on models, not training-seed uncertainty',
            'timing_scope': 'last logged updates only; excludes setup, checkpointing and evaluation; not billed runtime',
            'evidence_scope': 'report consistency against separate checkpoint audits; no weight reload or remote durability check'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'baselines', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('preserve previous comparison')
    result = summarize(args.root, args.baselines)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_report(args.out, result)
    print('RECOVERY_COMPARISON', result['ce_ranking'], flush=True)


if __name__ == '__main__':
    main()
