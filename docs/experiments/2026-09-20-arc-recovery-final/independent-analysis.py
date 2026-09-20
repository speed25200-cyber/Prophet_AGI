"""Recompute exported ARC summaries using original local items and plain arithmetic.

Run from the repository root. Requires NumPy and data/arc-recovery-eval-v1/items.jsonl.
No model forward is repeated; this checks identities and score aggregation.
"""
import hashlib
import gzip
import json
import math
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
ARMS = ['donor', 'hybrid-ce', 'attention-ce', 'hybrid-kl', 'attention-kl']


def digest(path):
    return hashlib.sha256(read_bytes(path)).hexdigest()


def read_bytes(path):
    if path.exists():
        return path.read_bytes()
    return gzip.decompress(path.with_suffix(path.suffix+'.gz').read_bytes())


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (a, b)


manifest = json.loads((ROOT/'items/manifest.json').read_bytes())
items_file = Path('data/arc-recovery-eval-v1/items.jsonl')
assert digest(items_file) == manifest['items_sha256']
items = [json.loads(line) for line in items_file.read_text(encoding='utf-8').splitlines()]
assert len(items) == 2376 and len({item['id'] for item in items}) == 2376
queue = json.loads((ROOT/'queue.json').read_bytes())
assert queue['status'] == 'all_five_complete'
assert queue['revision'] == '6d36dced81bfd3378279bcabe9cba9df5304f0be'
assert manifest == queue['input_manifest']
reports, correctness, summary = {}, {}, {}
runtime = None
for arm in ARMS:
    path = ROOT/(arm+'.json')
    assert digest(path) == queue['results'][arm]['report_sha256']
    r = json.loads(read_bytes(path))
    assert r['complete'] and r['protocol'] == 'arc-easy-raw-choice-evaluation-v1'
    assert r['manifest'] == manifest and r['manifest_sha256'] == digest(ROOT/'items/manifest.json')
    assert r['donor_revision'] == 'c1899de289a04d12100db370d81485cdf75e47ca'
    assert r['donor_weights_sha256'] == 'f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b'
    for key, source in [('script_sha256', 'scripts/eval_arc_recovery.py'),
                        ('scorer_sha256', 'prophet/eval/choices.py')]:
        blob = subprocess.check_output(['git', 'show', queue['revision']+':'+source])
        assert hashlib.sha256(blob).hexdigest() == r[key]
    if runtime is None:
        runtime = r['runtime']
    assert r['runtime'] == runtime
    assert runtime['device'] == 'cuda' and runtime['precision'] == 'float32'
    assert not runtime['allow_tf32_matmul'] and not runtime['allow_tf32_cudnn']
    if arm == 'donor':
        assert r['arm'] == 'donor' and r['checkpoint_audit'] is None and r['loop_k'] is None
    else:
        original = ROOT.parent/f'2026-09-20-qwen-recovery-{arm}-final'/(arm+'-final-audit.json')
        assert r['checkpoint_audit'] == json.loads(original.read_bytes())
        assert r['arm'] == 'recovered' and r['loop_k'] == 5
        assert r['config'] == r['checkpoint_audit']['config']
    e = r['evaluation']
    assert len(e['items']) == e['rows'] == 2376
    raw, normalized = [], []
    ties = norm_ties = gold_bytes = gold_tokens = 0
    gold_nats = chance = 0.0
    for row, item in zip(e['items'], items, strict=True):
        for field in ('id', 'source_row_sha256', 'gold'):
            assert row[field] == item[field]
        assert row['candidate_tokens'] == item['tokens']
        assert row['choice_characters'] == [len(c) for c in item['choices']]
        scores = row['choice_nats']
        assert len(scores) == len(item['choices']) and all(math.isfinite(s) and s >= 0 for s in scores)
        norm = [s/len(c) for s, c in zip(scores, item['choices'], strict=True)]
        a, b = min(range(len(scores)), key=scores.__getitem__), min(range(len(norm)), key=norm.__getitem__)
        assert row['prediction'] == a and row['prediction_character_normalized'] == b
        t, u = scores.count(scores[a]), norm.count(norm[b])
        assert row['ties'] == t and row['ties_character_normalized'] == u
        raw.append(a == row['gold'])
        normalized.append(b == row['gold'])
        ties += t > 1
        norm_ties += u > 1
        gold_nats += scores[row['gold']]
        gold_bytes += row['candidate_tokens'][row['gold']]['answer_bytes']
        gold_tokens += row['candidate_tokens'][row['gold']]['answer_tokens']
        chance += 1/len(scores)
    derived = {'accuracy': sum(raw)/len(items), 'accuracy_character_normalized': sum(normalized)/len(items),
               'uniform_choice_chance': chance/len(items), 'tied_items': ties,
               'tied_items_character_normalized': norm_ties,
               'gold_answer_total_nats': gold_nats, 'gold_answer_scored_bytes': gold_bytes,
               'gold_answer_scored_tokens': gold_tokens, 'gold_answer_nats_per_token': gold_nats/gold_tokens,
               'gold_answer_bits_per_byte': gold_nats/gold_bytes/math.log(2)}
    for key, value in derived.items():
        close(value, e[key])
    for key, value in queue['results'][arm].items():
        if key != 'report_sha256':
            close(value, derived[key])
    summary[arm] = dict(derived, correct=sum(raw), correct_character_normalized=sum(normalized),
                       report_sha256=digest(path), seconds=r['seconds'])
    correctness[arm] = {'raw': np.array(raw, dtype=np.int8), 'character_normalized': np.array(normalized, dtype=np.int8)}
    reports[arm] = r

contrasts = [('hybrid-ce','attention-ce'), ('hybrid-kl','attention-kl'),
             ('hybrid-kl','hybrid-ce'), ('attention-kl','attention-ce')]
contrasts += [(arm,'donor') for arm in ARMS[1:]]
paired = {}
for metric in ('raw', 'character_normalized'):
    paired[metric] = {}
    for left, right in contrasts:
        delta = correctness[left][metric]-correctness[right][metric]
        # Resampling counts of paired outcomes is exactly the empirical paired-item bootstrap.
        counts = np.array([(delta == x).sum() for x in (-1,0,1)])
        rng = np.random.default_rng(0)
        resampled = rng.multinomial(len(items), counts/len(items), size=10000)
        draws = (resampled[:,2]-resampled[:,0])/len(items)
        paired[metric][left+' minus '+right] = {
            'difference_percentage_points': float(delta.mean()*100),
            'ci95_percentage_points': (np.quantile(draws,[0.025,0.975])*100).tolist(),
            'left_only_correct': int(counts[2]), 'right_only_correct': int(counts[0]),
            'same_correctness': int(counts[1])}
output = {'complete': True, 'rows': len(items), 'input_items_sha256': digest(items_file),
          'analysis_script_sha256': digest(Path(__file__)), 'summary': summary, 'paired': paired,
          'bootstrap': {'replicates':10000,'seed':0,'method':'multinomial resampling of paired-item differences',
                        'scope':'95% unadjusted descriptive intervals; 8 contrasts per metric, one training seed. No seed or contamination uncertainty.'},
          'checks': 'Exact report/source/token/checkpoint/runtime identities and every per-choice ranking, tie and aggregate recomputed. No GPU forward reproduced.'}
(ROOT/'independent-verification.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({'summary':summary, 'paired_raw':paired['raw']},indent=2))
