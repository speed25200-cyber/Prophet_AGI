"""Reproduce the completed factorial comparison and verify cache report evidence.

Run from the repository root; this reads reports, not checkpoint tensors.
"""
import hashlib
import json
import math
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import torch
from prophet.data.donor_tokenizer import DonorByteTokenizer
from scripts.audit_recovery_cache_suite import select_prefixes
from scripts.recover_qwen import read_documents
from scripts.summarize_recovery_pilot import ARMS, summarize

base = Path('docs/experiments')
out = base / '2026-09-20-qwen-recovery-attention-kl-final'
read = lambda p: json.loads(p.read_bytes())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
sources = {arm: base / ('2026-09-20-qwen-recovery-' + arm + '-final') for arm in ARMS}
manifest = read(out / 'export-manifest.json')
for name, record in manifest.items():
    assert (out/name).stat().st_size == record['bytes']
    assert sha(out/name) == record['sha256']
with tempfile.TemporaryDirectory(dir='data') as temporary:
    root = Path(temporary)
    shutil.copyfile(out/'plan.json', root/'plan.json')
    for arm, source in sources.items():
        assert read(source/'plan.json') == read(out/'plan.json')
        shutil.copytree(source/(arm+'-seed0'), root/(arm+'-seed0'))
        shutil.copyfile(source/(arm+'-final-audit.json'), root/(arm+'-final-audit.json'))
    recomputed = summarize(root, base/'2026-09-20-qwen-colab-fp32/policy')
assert recomputed == read(out/'comparison/four-arm-comparison.json')

cpu_paths = {'hybrid-ce': sources['hybrid-ce']/'cache/chunk64.json',
             'attention-ce': sources['attention-ce']/'attention-ce-cache.json',
             'hybrid-kl': sources['hybrid-kl']/'hybrid-kl-cache.json',
             'attention-kl': out/'comparison/attention-kl-cache.json'}
validation = Path('data/qwen-recovery-v1/validation.jsonl')
fingerprint = read(cpu_paths['hybrid-ce'])['tokenizer']
tokenizer = DonorByteTokenizer(Path('data/donor-qwen3-0.6b/source/tokenizer.json'),
    eos_id=fingerprint['eos_id'], pad_id=fingerprint['pad_id'], vocab_size=fingerprint['vocab_size'])
assert tokenizer.fingerprint() == fingerprint
selected = select_prefixes(read_documents(validation), tokenizer, 4, [128,512])
expected = {(doc,length): hashlib.sha256(torch.tensor([ids[:length]], dtype=torch.long).numpy().tobytes()).hexdigest()
            for doc,ids in selected for length in (128,512)}
queue = read(out/'gpu-cache/queue.json')
assert queue['status'] == 'all_passed'
suite = ET.parse(out/'gpu-cache/cuda-suite-tests.xml').getroot().find('testsuite')
assert {k:int(suite.attrib[k]) for k in ('tests','errors','failures','skipped')} == {'tests':4,'errors':0,'failures':0,'skipped':0}
assert sum('[cuda-' in c.attrib['name'] for c in suite.findall('testcase')) == 2
cache_summary = {}
for arm in ARMS:
    audit = read(sources[arm]/(arm+'-final-audit.json'))
    cpu = read(cpu_paths[arm])
    gpu_path = out/'gpu-cache'/(arm+'.json')
    gpu = read(gpu_path)
    assert sha(gpu_path) == queue['results'][arm]['sha256']
    assert queue['results'][arm]['exit_code'] == 0
    assert queue['results'][arm]['checkpoint'] == audit['checkpoint']
    assert gpu['device'] == 'cuda' and gpu['device_name'] == 'NVIDIA A100-SXM4-40GB'
    assert gpu['gdn_scan'] == ('fla_chunk32' if arm.startswith('hybrid-') else None)
    assert gpu['allow_tf32_matmul'] is gpu['allow_tf32_cudnn'] is False
    assert cpu['device'] == 'cpu'
    for key in ('selection','documents_requested','lengths','validation_sha256','tokenizer','loop_k','atol','rtol','precision'):
        assert cpu[key] == gpu[key], (arm,key)
    assert gpu['atol'] == gpu['rtol'] == 1e-4 and gpu['precision'] == 'float32'
    assert sha(validation) == gpu['validation_sha256']
    for device,report in [('cpu',cpu),('cuda',gpu)]:
        assert report['complete'] and report['passed'] and len(report['cases']) == 8
        assert report['recovery_checkpoint_audit'] == audit
        errors, ratios, memory = [], [], {}
        for case in report['cases']:
            assert case['passed'] and set(case['paths']) == {'chunked_prefill_then_decode','tokenwise'}
            assert case['input_ids_sha256'] == expected[(case['document_sha256'],case['length'])]
            for path in case['paths'].values():
                assert path['passed'] and path['all_logits_finite']
                assert path['positions'] == path['argmax_matches'] == case['length']
                assert 0 <= path['max_tolerance_ratio'] <= 1
                assert path['cache']['position'] == case['length']
                memory.setdefault(case['length'],path['cache'])
                assert memory[case['length']] == path['cache']
                errors.append(path['max_absolute_error']); ratios.append(path['max_tolerance_ratio'])
        cache_summary[arm+'-'+device] = {'cases':8,'paths':len(errors),'maximum_logit_error':max(errors),
            'maximum_tolerance_ratio':max(ratios),'cache_by_length':memory,
            'execution_seconds':report['execution_seconds']}
    assert cache_summary[arm+'-cpu']['cache_by_length'] == cache_summary[arm+'-cuda']['cache_by_length']
    rows = [json.loads(line) for line in (sources[arm]/(arm+'-seed0/training.jsonl')).read_text().splitlines()]
    assert all(math.isfinite(v) for row in rows for v in row['extra'].values())
result = {'complete':True,'protocol':'four-arm-independent-report-verification-v1',
          'factorial_report_reproduced_exactly':True,'bootstrap_draws':10000,
          'comparison_sha256':sha(out/'comparison/four-arm-comparison.json'),
          'export_members_verified':len(manifest),'miniature_cases':4,'miniature_cuda_cases':2,
          'cache':cache_summary,
          'scope':'All four strict endpoint contracts and paired comparisons reproduced from downloaded reports. Cache input IDs independently rebuilt from corpus/tokenizer; all 64 CPU and 64 CUDA paths checked. No new tensor load, remote weight durability, seed robustness, long-context or capability claim.'}
(out/'independent-verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
print('FOUR_ARM_REPRODUCED',recomputed['ce_ranking'])
for key,value in cache_summary.items():
    print('CACHE',key,value['maximum_logit_error'],value['maximum_tolerance_ratio'])
