#!/usr/bin/env python3
"""Check an evaluated recovery checkpoint on deterministic development prefixes.

CPU FP32 only. This expands the original single-prefix diagnostic without
relaxing its tolerance or claiming coverage of CUDA/BF16/variable depth.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from prophet.config import ProphetConfig  # noqa: E402
from prophet.data.donor_tokenizer import DonorByteTokenizer  # noqa: E402
from prophet.modeling.layers import GatedDeltaNet  # noqa: E402
from prophet.modeling.model import ProphetCache, ProphetModel  # noqa: E402
from scripts.audit_recovery_checkpoint import load_evaluated_checkpoint, require  # noqa: E402
from scripts.recover_qwen import digest, read_documents, write_report  # noqa: E402


def select_prefixes(documents, tokenizer, count, lengths):
    """Select distinct eligible documents by text SHA256, before model execution."""
    require(count > 0 and lengths and min(lengths) >= 3, "invalid prefix selection")
    by_hash = {hashlib.sha256(text.encode('utf-8')).hexdigest(): text for text in documents}
    selected = []
    for document_sha, text in sorted(by_hash.items()):
        ids = tokenizer.encode(text, add_eos=False)
        if len(ids) >= max(lengths):
            selected.append((document_sha, ids[:max(lengths)]))
        if len(selected) == count:
            break
    require(len(selected) == count, "not enough distinct documents for every requested length")
    return selected


@torch.inference_mode()
def check_prefix(model, ids, loop_k):
    """Compare every logit at every position; fresh cache for each execution path."""
    length = ids.shape[1]
    require(ids.device.type == 'cpu' and ids.shape[0] == 1 and length >= 3, "invalid CPU prefix")
    full = model(ids, loop_k=loop_k, return_mtp=False).logits
    require(full.dtype == torch.float32, "cache suite requires FP32 logits")
    full_finite = bool(torch.isfinite(full).all())
    split = min(64, length // 2)
    paths = {}
    for name, chunks in [('chunked_prefill_then_decode', [split, length-split-1, 1]),
                         ('tokenwise', [1] * length)]:
        cache = ProphetCache()
        position, maximum, ratio, matches = 0, 0.0, 0.0, 0
        finite = full_finite
        for size in chunks:
            actual = model(ids[:, position:position+size], cache=cache,
                           loop_k=loop_k, return_mtp=False).logits
            expected = full[:, position:position+size]
            require(actual.shape == expected.shape and actual.dtype == torch.float32,
                    "cached output shape or precision differs")
            pair_finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
            finite = finite and pair_finite
            if pair_finite:
                error = (actual-expected).abs()
                maximum = max(maximum, error.max().item())
                ratio = max(ratio, (error / (1e-4 + 1e-4 * expected.abs())).max().item())
                matches += int((actual.argmax(-1) == expected.argmax(-1)).sum())
            position += size
        require(cache.position == position == length, "cache position differs")
        paths[name] = {'all_logits_finite': finite, 'positions': position,
                       'max_absolute_error': maximum if finite else None,
                       'max_tolerance_ratio': ratio if finite else None,
                       'argmax_matches': matches, 'passed': finite and ratio <= 1,
                       'cache': cache.summary()}
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'source', 'validation', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--documents', type=int, default=4)
    parser.add_argument('--lengths', type=int, nargs='+', default=[128, 512])
    parser.add_argument('--reference-scan', action='store_true')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('preserve existing cache evidence')
    require(args.documents > 0 and min(args.lengths) >= 3, 'invalid prefix selection')
    require(len(set(args.lengths)) == len(args.lengths), 'duplicate prefix lengths')
    torch.set_num_threads(2)
    state, audit = load_evaluated_checkpoint(args.run, args.step)
    contract = audit['training_contract']
    identity = contract['run_identity']
    require(max(args.lengths) <= contract['seq_len'], 'prefix exceeds the trained window')
    require(digest(args.validation) == identity['validation_sha256'], 'validation identity differs')
    fingerprint = identity['tokenizer']
    tokenizer = DonorByteTokenizer(args.source / 'tokenizer.json', eos_id=fingerprint['eos_id'],
                                   pad_id=fingerprint['pad_id'], vocab_size=fingerprint['vocab_size'])
    require(tokenizer.fingerprint() == fingerprint, 'tokenizer identity differs')
    selected = select_prefixes(read_documents(args.validation), tokenizer, args.documents, args.lengths)
    model = ProphetModel(ProphetConfig.from_dict(state['config']))
    model.load_state_dict(state['model'], strict=True)
    del state
    model.eval()
    gdn = [layer for layer in model.modules() if isinstance(layer, GatedDeltaNet)]
    if args.reference_scan:
        require(bool(gdn), 'reference scan requires GDN')
        for layer in gdn:
            layer.chunk_size = None
    result = {'protocol': 'recovered-development-cache-v1', 'complete': False,
              'scope': 'deterministic development prefixes, CPU FP32, fixed trained depth; not deployment certification',
              'selection': 'first distinct eligible document SHA256 values in ascending order; no EOS',
              'documents_requested': args.documents, 'lengths': args.lengths,
              'validation_sha256': identity['validation_sha256'], 'tokenizer': fingerprint,
              'recovery_checkpoint_audit': audit, 'torch': str(torch.__version__),
              'num_threads': torch.get_num_threads(), 'precision': 'float32', 'device': 'cpu',
              'loop_k': identity['evaluation_loop_k'], 'script_sha256': digest(Path(__file__)),
              'gdn_scan': ('sequential_reference' if args.reference_scan else 'chunk64') if gdn else None,
              'atol': 1e-4, 'rtol': 1e-4, 'cases': []}
    started = time.monotonic()
    for document_sha, tokens in selected:
        for length in args.lengths:
            ids = torch.tensor([tokens[:length]], dtype=torch.long)
            paths = check_prefix(model, ids, result['loop_k'])
            case = {'document_sha256': document_sha, 'length': length,
                    'input_ids_sha256': hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
                    'paths': paths, 'passed': all(p['passed'] for p in paths.values())}
            result['cases'].append(case)
            print('CACHE_SUITE_PROGRESS', len(result['cases']), document_sha, length, case['passed'], flush=True)
    result.update(complete=True, passed=all(case['passed'] for case in result['cases']),
                  execution_seconds=time.monotonic()-started)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_report(args.out, result)
    if not result['passed']:
        raise SystemExit('cache numerical tolerance failed; all requested cases retained')


if __name__ == '__main__':
    main()
