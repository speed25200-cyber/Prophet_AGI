import hashlib
import json
import time
from pathlib import Path

from prophet.data.corpus import LocalTextSource
from prophet.data.tokenizer import ProphetTokenizer

root = Path('data/fineweb-fresh-v1')
tokenizer_path = Path('data/fineweb-pilot-v1/tokenizer.json')
out = Path('data/fineweb-fresh-token-counts.json')
assert not out.exists()
manifest_bytes = (root / 'manifest.json').read_bytes()
manifest = json.loads(manifest_bytes)
assert manifest['complete']
tokenizer = ProphetTokenizer.load(tokenizer_path)
result = {
    'corpus_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
    'tokenizer_sha256': hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
    'eos_per_document': True,
    'splits': {},
    'scope': 'Token counting only; no model inference or validation-loss scoring.',
}
for split in ('train', 'validation'):
    source = LocalTextSource.from_root(root / split, 'fineweb-edu', 1.0)
    expected = [m['sha256'] for name, m in sorted(manifest['artifacts'].items())
                if name.startswith(split + '/')]
    assert source.fingerprint()['files'] == expected
    start = time.monotonic()
    tokens = sum(len(tokenizer.encode(text, add_eos=True)) for text in source.open())
    result['splits'][split] = {
        'documents': source.n_documents(), 'tokens': tokens,
        'fingerprint': source.fingerprint(), 'seconds': time.monotonic() - start,
    }
    print(split, result['splits'][split], flush=True)
out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
