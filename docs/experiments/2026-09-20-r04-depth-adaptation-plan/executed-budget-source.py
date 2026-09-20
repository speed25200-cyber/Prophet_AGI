import json
import os
import subprocess
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from prophet.config import ProphetConfig
from prophet.budget import count_parameters
from prophet.modeling.model import ProphetModel

out = Path('data/r04-depth-adaptation-plan-v2')
assert not out.exists() or not any(out.iterdir())
out.mkdir(exist_ok=True)
source = Path('docs/experiments/2026-09-19-r04-step4096/loop-seed0/evaluation-step-004096.json')
parent = json.loads(source.read_text())
configs = {}
for arm in ['fixed4', 'uniform2to6']:
    cfg = ProphetConfig.from_dict(parent['run_protocol']['config'])
    cfg.recurrent.train_loop_dist = 'uniform'
    cfg.recurrent.train_loop_min = 4 if arm == 'fixed4' else 2
    cfg.recurrent.train_loop_max = 4 if arm == 'fixed4' else 6
    cfg.recurrent.truncated_backprop_steps = 6
    cfg.validate()
    with torch.device('meta'):
        model = ProphetModel(cfg)
    assert count_parameters(cfg).total == sum(p.numel() for p in model.parameters()) == 374689648
    cfg.to_json(out / (arm + '.json'))
    configs[arm] = cfg.to_dict()
env = {**os.environ, 'PYTHONIOENCODING':'utf-8'}
commands = [(arm+'-budget.txt', ['-m','prophet.budget',str(out/(arm+'.json')),'--a100-hours','1']) for arm in configs]
worst = ProphetConfig.from_dict(configs['uniform2to6'])
worst.recurrent.default_loop_k = 6
worst.to_json(out/'worst-k6-sizing-only.json')
commands += [('worst-k6-budget.txt',['-m','prophet.budget',str(out/'worst-k6-sizing-only.json'),'--a100-hours','1'])]
commands += [('search.txt',['scripts/design_search.py','--a100-hours','1','--top','3']), ('plan.txt',['-m','prophet.plan','--a100-hours','1'])]
for name, command in commands:
    output = subprocess.check_output([sys.executable,*command],env=env)
    (out/name).write_bytes(output.replace(b'\r\n',b'\n'))
tokens = 512 * 8 * 2048
train_tokens = json.loads(Path('docs/experiments/2026-09-19-pilot-token-counts.json').read_text())['splits']['train']['tokens']
assert parent['train_tokens'] + tokens <= 4 * train_tokens
report = {
    'status':'budgeted_protocol_only_not_launched',
    'parent_checkpoint':parent['checkpoint'],
    'parameters_each':374689648,'steps_each':512,'batch_size':8,'seq_len':2048,
    'parameter_count_scope':'Budget agrees exactly with meta-device ProphetModel construction. Historical R04 tables used a pre-b015121 budget that omitted 1136 gate-bias and GDN output-normalization parameters; no weights are added by this correction.',
    'additional_tokens_each':tokens,'cumulative_tokens_each':parent['train_tokens']+tokens,
    'original_training_corpus_tokens':train_tokens,
    'cumulative_corpus_passes_each':(parent['train_tokens']+tokens)/train_tokens,
    'mean_training_loops':{'fixed4':4,'uniform2to6':4},
    'maximum_backpropagated_loops':6,
    'nominal_total_gpu_hours':1,'hard_queue_seconds':5400,
    'nominal_paired_training_seconds_at_2_seconds_per_step':2*512*2,
    'timing_scope':'Projection from measured fixed-k4 R04 steps; variable-depth, warm-start, evaluation and checkpoint costs require measurement.',
    'budget_scope':'These CLI reports describe full training candidates, not measured warm-start memory or training viability. Per-arm worst-case k6 CUDA preflight is required; design-search from-scratch feasibility is not a warm-start result.',
    'recipe':{'muon_lr':0.001,'adamw_lr':0.00003,'weight_decay':0.1,'grad_clip':1.0,'schedule':'512-step WSD with existing 100-step minimum warmup and 18% decay','precision':'original R04 BF16 autocast with FP32 master weights; exact runtime and GPU gate required'},
    'warm_start':'Same final weights and loader cursor in both arms; new optimizer and LR schedule. Not an exact continuation of the original optimizer. Each new arm must resume exactly within its own frozen contract.',
    'primary_gate':'At step512, uniform arm k6/k4 BPB <=0.995 with paired-document 95% bootstrap upper difference <0, and uniform k4 BPB <=1.01*fixed4 k4 BPB. Must also improve k6 versus the matched fixed4 arm. Seed0 screening only; repeat seeds before adoption.',
    'stop':'No long run on nonfinite updates, changed provenance, failed k4 baseline reproduction, failed actual CUDA resume or peak memory gate. Preserve partial reports; no recipe tuning against development scores.',
    'scope':'Training-depth policy ablation; no model topology changes, adaptive halting, reasoning-capability or device deployment claim.'
}
(out/'protocol.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(report,indent=2))
