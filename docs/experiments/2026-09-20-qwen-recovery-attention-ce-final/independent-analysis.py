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
out = base/'2026-09-20-qwen-recovery-attention-ce-final'
old = base/'2026-09-20-qwen-recovery-hybrid-ce-final'
baselines = base/'2026-09-20-qwen-colab-fp32/policy'
read = lambda p: json.loads(p.read_bytes())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
with tempfile.TemporaryDirectory(dir='data') as temporary:
    root = Path(temporary)
    shutil.copy2(out/'plan.json', root/'plan.json')
    for arm, source in [('hybrid-ce', old), ('attention-ce', out)]:
        shutil.copytree(source/(arm+'-seed0'), root/(arm+'-seed0'))
        shutil.copy2(source/(arm+'-final-audit.json'), root/(arm+'-final-audit.json'))
    try:
        summarize(root, baselines, draws=100)
    except FileNotFoundError as exc:
        assert Path(exc.filename) == root/'hybrid-kl-seed0/evaluation-step-002048.json'
    else:
        raise AssertionError('Full comparator must refuse the unfinished four-arm pilot')

reports = {arm: read(source/(arm+'-seed0/evaluation-step-002048.json'))
           for arm, source in [('hybrid-ce', old), ('attention-ce', out)]}
audits = {arm: read(source/(arm+'-final-audit.json'))
          for arm, source in [('hybrid-ce', old), ('attention-ce', out)]}
assert {k:v for k,v in reports['hybrid-ce']['identity'].items() if k not in ('initialization_sha256','objective')} == {k:v for k,v in reports['attention-ce']['identity'].items() if k not in ('initialization_sha256','objective')}
assert {k:v for k,v in audits['hybrid-ce']['training_contract'].items() if k not in ('run_identity','distillation')} == {k:v for k,v in audits['attention-ce']['training_contract'].items() if k not in ('run_identity','distillation')}
cache = read(out/'attention-ce-cache.json')
hybrid_cache = read(old/'cache/chunk64.json')
assert cache['recovery_checkpoint_audit'] == audits['attention-ce']
assert sha(out/'attention-ce-final-audit.json') == '16f3b3f64a01bbe475571bd965109b3787390779f41bc261e45359d8529070a9'
assert sha(out/'attention-ce-cache.json') == 'f5f10b39cca64884e5df7691402f640be4e5378c041efab5f37fd7a5ca2c622c'
assert cache['complete'] and cache['passed'] and len(cache['cases']) == 8
for key in ('selection','documents_requested','lengths','validation_sha256','tokenizer','torch','num_threads','precision','device','loop_k','atol','rtol'):
    assert cache[key] == hybrid_cache[key], key
validation = Path('data/qwen-recovery-v1/validation.jsonl')
assert sha(validation) == cache['validation_sha256']
fingerprint = cache['tokenizer']
tokenizer = DonorByteTokenizer(Path('data/donor-qwen3-0.6b/source/tokenizer.json'), eos_id=fingerprint['eos_id'], pad_id=fingerprint['pad_id'], vocab_size=fingerprint['vocab_size'])
assert tokenizer.fingerprint() == fingerprint
selected = select_prefixes(read_documents(validation), tokenizer, 4, [128,512])
expected = {(doc,n): hashlib.sha256(torch.tensor([ids[:n]], dtype=torch.long).numpy().tobytes()).hexdigest() for doc,ids in selected for n in [128,512]}
memory = {}
maximum_error = maximum_ratio = 0
for a,b in zip(cache['cases'], hybrid_cache['cases'], strict=True):
    for key in ('document_sha256','length','input_ids_sha256'):
        assert a[key] == b[key]
    assert a['input_ids_sha256'] == expected[(a['document_sha256'], a['length'])]
    assert a['passed'] and set(a['paths']) == {'chunked_prefill_then_decode','tokenwise'}
    for path in a['paths'].values():
        assert path['passed'] and path['all_logits_finite'] and path['argmax_matches'] == path['positions'] == a['length']
        assert path['max_tolerance_ratio'] <= 1
        maximum_error = max(maximum_error, path['max_absolute_error'])
        maximum_ratio = max(maximum_ratio, path['max_tolerance_ratio'])
        row = {'attention':path['cache'], 'hybrid':b['paths']['tokenwise']['cache']}
        assert memory.setdefault(a['length'],row) == row
rows = [json.loads(line) for line in (out/'attention-ce-seed0/training.jsonl').read_text().splitlines()]
assert all(math.isfinite(value) for row in rows for value in row['extra'].values())
attention = reports['attention-ce']['evaluation']
initial = read(baselines/'baseline-attention.json')['evaluation']
donor = read(baselines/'baseline-donor.json')['evaluation']
result = {
 'protocol':'two-completed-ce-arms-interim-analysis-v1',
 'attention_versus_initialization':paired_document_difference(attention,initial),
 'hybrid_minus_attention':paired_document_difference(reports['hybrid-ce']['evaluation'],attention),
 'attention_fraction_initial_ce_gap_closed':(initial['nats_per_token']-attention['nats_per_token'])/(initial['nats_per_token']-donor['nats_per_token']),
 'attention_median_logged_update_seconds_last_256':statistics.median(row['seconds'] for row in rows[-256:]),
 'attention_cache':{'cases':8,'paths':16,'maximum_logit_error':maximum_error,'maximum_tolerance_ratio':maximum_ratio,'execution_seconds':cache['execution_seconds']},
 'measured_cache_bytes_by_prefix_length':memory,
 'archive':{'bytes':171906,'sha256':'73fb41313b1fc28b2f88adbe3bf84e6854e43211007b410a81bd2848dc8715e6'},
 'scope':'Interim CE-only seed-zero comparison; both endpoint reports pass full-comparator checks before its required refusal at the missing third arm. Conditional document intervals exclude seed uncertainty. Cache storage excludes weights, workspace and peak allocation. No fresh checkpoint reload or remote durability check; attention weights remain local to Colab.'
}
(out/'paired-and-cache-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
