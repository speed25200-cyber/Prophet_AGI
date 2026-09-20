#!/usr/bin/env python3
"""Score the unchanged donor or an exact recovered checkpoint on frozen ARC-Easy.

Development capability probe, independent of language-loss selection. No training,
chat adaptation, reserved Tier-2 benchmarks, generation or test-set filtering.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from prophet.config import ProphetConfig  # noqa: E402
from prophet.eval.choices import (  # noqa: E402
    continuation_nats,
    encode_choice,
    rank_choices,
    token_ids_sha256,
)
from prophet.modeling.layers import HAS_FLA  # noqa: E402
from prophet.modeling.model import ProphetModel  # noqa: E402
from scripts.audit_recovery_checkpoint import load_evaluated_checkpoint, require  # noqa: E402
from scripts.eval_qwen_recovery import DonorEvaluationAdapter  # noqa: E402
from scripts.prepare_arc_recovery_eval import SOURCE  # noqa: E402
from scripts.recover_qwen import (  # noqa: E402
    REVISION,
    WEIGHTS_SHA256,
    digest,
    load_source,
    recovery_precision,
    write_report,
)


def encoded_items(items, tokenizer, *, max_tokens):
    """Validate every candidate before any model is loaded; preserve source order."""
    seen, result = set(), []
    for item in items:
        require(item['id'] not in seen, 'duplicate item')
        seen.add(item['id'])
        require(2 <= len(item['choices']) <= 5 and
                len(item['choices']) == len(item['tokens']), 'invalid choices')
        require(type(item['gold']) is int and 0 <= item['gold'] < len(item['choices']), 'invalid gold index')
        candidates = []
        for choice, expected in zip(item['choices'], item['tokens'], strict=True):
            ids, start = encode_choice(tokenizer, item['prompt'], choice, max_tokens=max_tokens)
            actual = {'input_ids_sha256': token_ids_sha256(ids), 'tokens': len(ids),
                      'context_tokens': start, 'answer_tokens': len(ids)-start,
                      'answer_bytes': tokenizer.byte_length(ids[start:])}
            require(actual == expected, 'candidate tokenization changed')
            require(actual['answer_bytes'] > 0, 'empty answer payload')
            candidates.append((ids, start))
        result.append((item, candidates))
    require(bool(result), 'empty evaluation')
    return result


def evaluate_items(model, encoded, *, device='cpu', loop_k=None, progress=None):
    model.eval()
    rows = []
    for index, (item, candidates) in enumerate(encoded):
        nats = [continuation_nats(model, ids, start, device=device, loop_k=loop_k)
                for ids, start in candidates]
        ranking = rank_choices(nats, [len(choice) for choice in item['choices']])
        rows.append({'id': item['id'], 'source_row_sha256': item['source_row_sha256'],
                     'gold': item['gold'], 'choice_nats': nats,
                     'choice_characters': [len(choice) for choice in item['choices']],
                     'candidate_tokens': item['tokens'], **ranking})
        if progress is not None:
            progress(index+1, len(encoded))
    gold_nats = sum(row['choice_nats'][row['gold']] for row in rows)
    gold_bytes = sum(row['candidate_tokens'][row['gold']]['answer_bytes'] for row in rows)
    gold_tokens = sum(row['candidate_tokens'][row['gold']]['answer_tokens'] for row in rows)
    return {'items': rows, 'rows': len(rows),
            'accuracy': sum(row['prediction'] == row['gold'] for row in rows)/len(rows),
            'accuracy_character_normalized': sum(row['prediction_character_normalized'] == row['gold'] for row in rows)/len(rows),
            'uniform_choice_chance': sum(1/len(row['choice_nats']) for row in rows)/len(rows),
            'tied_items': sum(row['ties'] > 1 for row in rows),
            'tied_items_character_normalized': sum(row['ties_character_normalized'] > 1 for row in rows),
            'gold_answer_total_nats': gold_nats, 'gold_answer_scored_tokens': gold_tokens,
            'gold_answer_scored_bytes': gold_bytes, 'gold_answer_nats_per_token': gold_nats/gold_tokens,
            'gold_answer_bits_per_byte': gold_nats/gold_bytes/math.log(2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'items', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--arm', choices=('donor', 'recovered'), required=True)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--step', type=int)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if args.arm == 'recovered' and (args.run is None or args.step is None):
        parser.error('recovered evaluation requires --run and --step')
    if args.arm == 'donor' and (args.run is not None or args.step is not None):
        parser.error('donor evaluation does not take checkpoint arguments')
    if args.out.exists():
        raise FileExistsError('preserve previous capability results')
    require(args.device != 'cuda' or torch.cuda.is_available(), 'requested CUDA is unavailable')
    torch.set_num_threads(2)
    torch.manual_seed(0)
    device = torch.device(args.device)
    recovery_precision('float32', device)
    source_config, tokenizer = load_source(args.source)
    manifest_path = args.items/'manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    require(manifest['complete'] and manifest['protocol'] == 'arc-easy-raw-choice-v1' and
            manifest['source'] == SOURCE and manifest['rows'] == 2376 and manifest['max_tokens'] == 512,
            'incompatible frozen evaluation manifest')
    require(manifest['tokenizer'] == tokenizer.fingerprint() and
            manifest['items_sha256'] == digest(args.items/'items.jsonl'), 'evaluation inputs changed')
    items = [json.loads(line) for line in (args.items/'items.jsonl').read_text(encoding='utf-8').splitlines()]
    require(len(items) == manifest['rows'], 'evaluation item count changed')
    encoded = encoded_items(items, tokenizer, max_tokens=manifest['max_tokens'])
    require(sum(len(ids) for _, candidates in encoded for ids, _ in candidates) ==
            manifest['total_candidate_tokens'], 'candidate token total changed')
    import transformers
    from transformers import AutoModelForCausalLM
    audit, config, loop_k = None, None, None
    if args.arm == 'donor':
        model = DonorEvaluationAdapter(AutoModelForCausalLM.from_pretrained(
            args.source, local_files_only=True, trust_remote_code=False,
            dtype=torch.float32, attn_implementation='sdpa'))
    else:
        state, audit = load_evaluated_checkpoint(args.run, args.step)
        identity = audit['training_contract']['run_identity']
        require(identity['tokenizer'] == tokenizer.fingerprint() and
                identity['donor_revision'] == REVISION and identity['donor_weights_sha256'] == WEIGHTS_SHA256
                and identity['donor_config'] == source_config, 'checkpoint donor/tokenizer identity differs')
        require(audit['training_contract']['dtype'] == 'float32' and
                audit['training_contract']['allow_tf32'] is False, 'checkpoint precision policy differs')
        require(manifest['max_tokens'] <= audit['training_contract']['seq_len'], 'evaluation exceeds trained window')
        config = ProphetConfig.from_dict(audit['config'])
        if device.type == 'cuda' and 'gdn' in config.recurrent.core_pattern:
            require(HAS_FLA, 'CUDA hybrid evaluation requires FLA')
        loop_k = identity['evaluation_loop_k']
        model = ProphetModel(config)
        model.load_state_dict(state['model'], strict=True)
        config = config.to_dict()
        del state
    model.to(device).eval()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    def progress(done, total):
        if done % 100 == 0 or done == total:
            print('ARC_PROGRESS', done, total, round(time.perf_counter()-started, 2), flush=True)

    evaluation = evaluate_items(model, encoded, device=device, loop_k=loop_k, progress=progress)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    report = {'complete': True, 'protocol': 'arc-easy-raw-choice-evaluation-v1', 'arm': args.arm,
              'manifest': manifest, 'manifest_sha256': digest(manifest_path),
              'donor_revision': REVISION, 'donor_weights_sha256': WEIGHTS_SHA256,
              'source_config': source_config, 'checkpoint_audit': audit, 'config': config,
              'loop_k': loop_k, 'evaluation': evaluation, 'seconds': time.perf_counter()-started,
              'script_sha256': digest(Path(__file__)),
              'scorer_sha256': digest(Path(__file__).resolve().parent.parent/'prophet/eval/choices.py'),
              'runtime': {'device': str(device), 'torch': str(torch.__version__),
                          'transformers': transformers.__version__, 'cuda': torch.version.cuda,
                          'fla': version('fla-core') if device.type == 'cuda' and HAS_FLA else None,
                          'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
                          'precision': 'float32', 'batch_size': 1, 'cpu_threads': 2,
                          'allow_tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
                          'allow_tf32_cudnn': torch.backends.cudnn.allow_tf32,
                          'triton_f32_default': os.environ.get('TRITON_F32_DEFAULT')},
              'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
              'scope': 'Full pinned ARC-Easy test, zero-shot raw-text continuation likelihood. Exploratory capability evaluation, not an instruction-following or AGI claim; donor contamination unknown. All candidates scored without truncation or cache.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_report(args.out, report)
    print('ARC_COMPLETE', args.arm, evaluation['accuracy'], evaluation['accuracy_character_normalized'],
          evaluation['gold_answer_bits_per_byte'], flush=True)


if __name__ == '__main__':
    main()
