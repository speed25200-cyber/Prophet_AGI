"""Reproduce the three-endpoint interim analysis from the repository root."""
import hashlib
import json
import math
import shutil
import statistics
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import torch
from prophet.data.donor_tokenizer import DonorByteTokenizer
from prophet.eval.paired import paired_document_difference
from scripts.audit_recovery_cache_suite import select_prefixes
from scripts.recover_qwen import read_documents
from scripts.summarize_recovery_pilot import summarize

base = Path('docs/experiments')
out = base / '2026-09-20-qwen-recovery-hybrid-kl-final'
baselines = base / '2026-09-20-qwen-colab-fp32/policy'
sources = {arm: base / ('2026-09-20-qwen-recovery-' + arm + '-final')
           for arm in ('hybrid-ce', 'attention-ce', 'hybrid-kl')}
read = lambda p: json.loads(p.read_bytes())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
plan = read(out / 'plan.json')
with tempfile.TemporaryDirectory(dir='data') as directory:
    root = Path(directory)
    shutil.copyfile(out / 'plan.json', root / 'plan.json')
    for arm, source in sources.items():
        assert read(source / 'plan.json') == plan
        shutil.copytree(source / (arm + '-seed0'), root / (arm + '-seed0'))
        shutil.copyfile(source / (arm + '-final-audit.json'), root / (arm + '-final-audit.json'))
    try:
        summarize(root, baselines, draws=100)
    except FileNotFoundError as exc:
        assert Path(exc.filename) == root / 'attention-kl-seed0/evaluation-step-002048.json'
    else:
        raise AssertionError('Four-arm publication requires the fourth endpoint')

reports = {arm: read(source / (arm + '-seed0/evaluation-step-002048.json'))
           for arm, source in sources.items()}
audits = {arm: read(source / (arm + '-final-audit.json')) for arm, source in sources.items()}
normalized_identities = [{k:v for k,v in r['identity'].items()
                          if k not in ('initialization_sha256', 'objective')}
                         for r in reports.values()]
normalized_contracts = [{k:v for k,v in a['training_contract'].items()
                         if k not in ('run_identity', 'distillation')}
                        for a in audits.values()]
assert all(x == normalized_identities[0] for x in normalized_identities)
assert all(x == normalized_contracts[0] for x in normalized_contracts)
assert audits['hybrid-kl']['config'] == audits['hybrid-ce']['config']
assert reports['hybrid-kl']['identity']['initialization_sha256'] == reports['hybrid-ce']['identity']['initialization_sha256']
assert sha(out / 'hybrid-kl-final-audit.json') == '70376ee369f770bdbb60250dd6f690c0d6b6012109afb246f7fa7980bc89aded'
assert sha(out / 'hybrid-kl-seed0/evaluation-step-002048.json') == '9d2e2563bb464d8580d02ebaaf97895d881d956c8720d55744b37f4213787a17'
assert sha(out / 'hybrid-kl-cache.json') == '6cb07e1281a463bb82611f6c166a26fe1f8fdec46d842c9c9c8f89221e19da6f'
record = read(out / 'endpoint-queue-record.json')
assert record['checkpoint'] == audits['hybrid-kl']['checkpoint']
assert record['audit_sha256'] == sha(out / 'hybrid-kl-final-audit.json')
assert record['evaluation_sha256'] == sha(out / 'hybrid-kl-seed0/evaluation-step-002048.json')
export = read(out / 'source-export-verification.json')
assert export['complete'] and export['matches_downloaded_archive']
assert export['bytes'] == 225462 and export['sha256'] == '10df8cf98212574a4897b318e6f32c6f0b8cb211bb42fa1de6456d95de6da334'
cache = read(out / 'hybrid-kl-cache.json')
old_cache = read(sources['hybrid-ce'] / 'cache/chunk64.json')
assert cache['recovery_checkpoint_audit'] == audits['hybrid-kl']
assert cache['complete'] and cache['passed'] and len(cache['cases']) == 8
for key in ('selection', 'documents_requested', 'lengths', 'validation_sha256', 'tokenizer',
            'torch', 'num_threads', 'precision', 'device', 'loop_k', 'gdn_scan', 'atol', 'rtol'):
    assert cache[key] == old_cache[key], key
validation = Path('data/qwen-recovery-v1/validation.jsonl')
assert sha(validation) == cache['validation_sha256']
fingerprint = cache['tokenizer']
tokenizer = DonorByteTokenizer(Path('data/donor-qwen3-0.6b/source/tokenizer.json'),
    eos_id=fingerprint['eos_id'], pad_id=fingerprint['pad_id'], vocab_size=fingerprint['vocab_size'])
assert tokenizer.fingerprint() == fingerprint
selected = select_prefixes(read_documents(validation), tokenizer, 4, [128, 512])
expected = {(doc, length): hashlib.sha256(torch.tensor([ids[:length]], dtype=torch.long).numpy().tobytes()).hexdigest()
            for doc, ids in selected for length in (128, 512)}
errors, ratios = [], []
for case, previous in zip(cache['cases'], old_cache['cases'], strict=True):
    assert case['passed'] and set(case['paths']) == {'chunked_prefill_then_decode', 'tokenwise'}
    assert case['input_ids_sha256'] == expected[(case['document_sha256'], case['length'])]
    assert all(case[k] == previous[k] for k in ('document_sha256', 'length', 'input_ids_sha256'))
    for name, path in case['paths'].items():
        assert path['passed'] and path['all_logits_finite']
        assert path['positions'] == path['argmax_matches'] == case['length']
        assert path['max_tolerance_ratio'] <= 1
        assert path['cache'] == previous['paths'][name]['cache']
        errors.append(path['max_absolute_error'])
        ratios.append(path['max_tolerance_ratio'])
rows = [json.loads(line) for line in (out / 'hybrid-kl-seed0/training.jsonl').read_text().splitlines()]
assert len(rows) == 2048
assert all(math.isfinite(value) for row in rows for value in row['extra'].values())
evaluation = reports['hybrid-kl']['evaluation']
initial = read(baselines / 'baseline-hybrid.json')['evaluation']
donor = read(baselines / 'baseline-donor.json')['evaluation']
result = {
    'protocol': 'three-completed-arms-interim-analysis-v1',
    'hybrid_kl_minus_hybrid_ce': paired_document_difference(evaluation, reports['hybrid-ce']['evaluation']),
    'hybrid_kl_minus_initialization': paired_document_difference(evaluation, initial),
    'hybrid_kl_fraction_initial_ce_gap_closed': (initial['nats_per_token'] - evaluation['nats_per_token']) / (initial['nats_per_token'] - donor['nats_per_token']),
    'student_tokens': reports['hybrid-kl']['tokens_seen'],
    'teacher_tokens': reports['hybrid-kl']['teacher_tokens_seen'],
    'median_logged_update_seconds_last_256': statistics.median(row['seconds'] for row in rows[-256:]),
    'cpu_cache': {'cases': len(cache['cases']), 'paths': len(errors), 'maximum_logit_error': max(errors),
                  'maximum_tolerance_ratio': max(ratios), 'execution_seconds': cache['execution_seconds']},
    'archive': {'bytes': 225462, 'sha256': '10df8cf98212574a4897b318e6f32c6f0b8cb211bb42fa1de6456d95de6da334',
                'verification': 'Downloaded archive size and SHA256 match the source notebook read in cell 268 after reconnection. Checkpoint-audit, evaluation and cache report hashes also match the live Colab queue.'},
    'scope': 'Single-seed within-hybrid comparison at equal student tokens. KL adds teacher compute. All three endpoint reports pass the strict comparator before its required refusal at the absent attention-KL report. Cross-arm normalized inputs/contracts also match independently. Conditional document intervals exclude seed uncertainty. No architecture adoption, new weight reload, GPU cache result or capability score. Hybrid-KL weights remain on Colab.'
}
(out / 'paired-and-cache-comparison.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
print(json.dumps(result, indent=2))
