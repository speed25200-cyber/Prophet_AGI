"""Two fixed local completions; an integration check, not a capability benchmark."""
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import torch
from prophet.config import ProphetConfig
from prophet.data.donor_tokenizer import DonorByteTokenizer
from prophet.modeling.model import ProphetModel
from scripts.audit_recovery_checkpoint import load_evaluated_checkpoint

output = Path('data/local-attention-generation-check.json')
assert not output.exists()
torch.set_num_threads(2)
torch.manual_seed(0)
state, audit = load_evaluated_checkpoint(Path('data/recovered-models/attention-ce-seed0'), 2048)
identity = audit['training_contract']['run_identity']
fingerprint = identity['tokenizer']
tokenizer = DonorByteTokenizer(Path('data/donor-qwen3-0.6b/source/tokenizer.json'),
    eos_id=fingerprint['eos_id'], pad_id=fingerprint['pad_id'], vocab_size=fingerprint['vocab_size'])
assert tokenizer.fingerprint() == fingerprint
model = ProphetModel(ProphetConfig.from_dict(state['config']))
model.load_state_dict(state['model'], strict=True)
del state
model.eval()
prompts = ['Once upon a time, in a small village,', 'Dans un petit village, une jeune fille']
result = {'protocol': 'local-recovered-generation-integration-v1', 'complete': False,
          'checkpoint': audit['checkpoint'], 'torch': str(torch.__version__), 'device': 'cpu',
          'precision': 'float32', 'cpu_threads': 2, 'loop_k': identity['evaluation_loop_k'],
          'prompts': prompts, 'new_tokens': 32, 'temperature': 0,
          'decoding_policy': 'Raw text, no chat template, greedy full-vocabulary generate(), fixed 32 tokens without EOS early stopping.',
          'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'samples': [],
          'scope': 'Two hand-written prompts fixed before execution. Local loading/tokenization/cached-generation integration and comparison of every emitted token with full-forward greedy predictions. No capability score, timing benchmark, instruction-following or deployment claim; no prompt selection by outcome.'}
started = time.monotonic()
with torch.inference_mode():
    for prompt in prompts:
        ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
        assert ids.shape[1] + 32 <= audit['training_contract']['seq_len']
        generated = model.generate(ids, max_new_tokens=32, loop_k=result['loop_k'], temperature=0)
        continuation = generated[0, ids.shape[1]:].tolist()
        reference = model(generated[:, :-1], return_mtp=False, loop_k=result['loop_k']).logits[:, ids.shape[1]-1:, :]
        assert reference.shape[1] == 32
        finite = bool(torch.isfinite(reference).all())
        matched = int((reference.argmax(-1)[0] == generated[0, ids.shape[1]:]).sum())
        sample = {'prompt': prompt, 'prompt_ids': ids[0].tolist(), 'generated_ids': continuation,
                  'continuation': tokenizer.decode(continuation), 'generated_tokens': len(continuation),
                  'full_forward_logits_finite': finite, 'greedy_token_matches_full_forward': matched,
                  'passed': finite and matched == 32}
        result['samples'].append(sample)
        print('GENERATION_SAMPLE', json.dumps(sample, ensure_ascii=True), flush=True)
result.update(complete=True, passed=all(x['passed'] for x in result['samples']), seconds=time.monotonic()-started)
output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8', newline='\n')
print('LOCAL_GENERATION_COMPLETE', result['passed'], round(result['seconds'], 2), flush=True)
