"""Disposable R04 preflight instrumentation; never resumes a training output."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--model-repo', type=Path, required=True)
parser.add_argument('--policy', choices=['strict', 'legacy', 'cudnn'], required=True)
parser.add_argument('--trace', type=Path, required=True)
args, remaining = parser.parse_known_args()
assert '--mode' in remaining and remaining[remaining.index('--mode') + 1] == 'preflight'
assert not args.trace.exists()
sys.path.insert(0, str(args.model_repo))
import scripts.adapt_r04_depth as driver
from prophet.train.distillation import state_sha256

torch.use_deterministic_algorithms(args.policy == 'strict')
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = args.policy in ('strict', 'cudnn')
original_profile = driver.profile_max_depth
trace = {'policy': args.policy, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'steps': []}

def save():
    args.trace.write_text(json.dumps(trace, indent=2) + '
')

def rng():
    return state_sha256({'cpu': torch.get_rng_state(), **{f'cuda-{i}': t for i, t in enumerate(torch.cuda.get_rng_state_all())}})

def instrument(trainer, *, steps=3):
    batch = trainer._batch
    clip = torch.nn.utils.clip_grad_norm_
    trace['initial_weights'] = state_sha256(trainer.model)
    trace['deterministic_algorithms'] = torch.are_deterministic_algorithms_enabled()
    trace['cudnn_deterministic'] = torch.backends.cudnn.deterministic
    def traced_batch():
        ids = batch()
        trace['steps'].append({'inputs': state_sha256({'ids': ids}), 'rng_before_forward': rng()})
        save()
        return ids
    def traced_clip(parameters, *a, **kw):
        row = trace['steps'][-1]
        row['gradients'] = {n: state_sha256({n: p.grad}) if p.grad is not None else None for n, p in trainer.model.named_parameters()}
        row['rng_after_backward'] = rng()
        save()
        return clip(parameters, *a, **kw)
    trainer._batch = traced_batch
    torch.nn.utils.clip_grad_norm_ = traced_clip
    try:
        result = original_profile(trainer, steps=steps)
        trace['result'] = result
        trace['final_weights'] = {n: state_sha256({n: p}) for n, p in trainer.model.state_dict().items()}
        trace['complete'] = True
        save()
        return result
    finally:
        trainer._batch = batch
        torch.nn.utils.clip_grad_norm_ = clip

driver.profile_max_depth = instrument
sys.argv = [sys.argv[0], *remaining]
save()
driver.main()
