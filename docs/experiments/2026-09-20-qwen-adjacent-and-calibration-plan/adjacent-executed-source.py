"""Fixed CPU follow-up: neighboring-depth sharing at equal registered parameter count.

Research-only script; no architecture adoption, cache claim or GPU training.
"""
import hashlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path.cwd()))
from scripts.probe_qwen_relaxed import PROJECTIONS, score_prefixes
from scripts.audit_recovery_cache_suite import select_prefixes
from scripts.recover_qwen import load_source, read_documents, digest, REVISION, WEIGHTS_SHA256


@contextmanager
def pair_sharing(layers, mode):
    assert len(layers) == 28 and mode in ('mean','first','last')
    restore = []
    try:
        for first in range(4,24,2):
            for path in PROJECTIONS:
                pair = [layers[i].get_submodule(path) for i in (first,first+1)]
                assert all(isinstance(m,nn.Linear) and m.bias is None for m in pair)
                if mode == 'mean':
                    weight = (pair[0].weight.detach()+pair[1].weight.detach())/2
                    shared = nn.Linear(weight.shape[1],weight.shape[0],bias=False)
                    shared.weight = nn.Parameter(weight)
                else:
                    shared = pair[mode == 'last']
                for i, original in zip((first,first+1),pair,strict=True):
                    parent, name = path.rsplit('.',1)
                    owner = layers[i].get_submodule(parent)
                    restore.append((owner,name,original))
                    setattr(owner,name,shared)
                assert layers[first].get_submodule(path) is layers[first+1].get_submodule(path)
        yield
    finally:
        for owner,name,module in restore:
            setattr(owner,name,module)


torch.set_num_threads(2)
torch.manual_seed(0)
source=Path('data/donor-qwen3-0.6b/source')
validation=Path('data/qwen-recovery-v1/validation.jsonl')
out=Path('data/qwen-adjacent-prefix-probe-v1')
assert not out.exists()
config,tokenizer=load_source(source)
assert digest(validation)=='06903e9925fb4e2ceed27eedc09f5d4b33b32e3e2ce1bfe6e1deed62a491e8af'
prefixes=[(sha,ids[:512]) for sha,ids in select_prefixes(read_documents(validation),tokenizer,16,[512])]
variants=['mean','first','last']
report={'complete':False,'protocol':'qwen-adjacent-pairs-prefix-v1',
        'script_sha256':digest(Path(__file__)),
        'scoring_script_sha256':digest(Path('scripts/probe_qwen_relaxed.py')),
        'donor_revision':REVISION,'donor_weights_sha256':WEIGHTS_SHA256,
        'source_config':config,'tokenizer':tokenizer.fingerprint(),
        'validation_sha256':digest(validation),
        'variants_declared_before_scores':variants,
        'expected_registered_parameters':438763520,
        'parameter_arithmetic':'596049920 - 10 * 15728640; 10 adjacent pairs share seven middle-layer projections, all 28 sets of original norms retained.',
        'budget_scope':'CPU-only diagnostic after archived budget/search/plan; original donor retained in memory for restoration. Registered size is not process memory. All candidates below 500M; no GPU allocation.',
        'selection':'First 16 distinct eligible development-document SHA256 values ascending; first 512 tokens, no EOS.',
        'mapping':[[i,i+1] for i in range(4,24,2)],
        'runtime':{'torch':str(torch.__version__),'device':'cpu','precision':'float32','threads':2,'attention':'sdpa'},
        'scope':'Untrained native serial donor scout. No Prophet auxiliary heads, GDN, reinjection, cache, training, variable depth, benchmark tuning or adoption.',
        'results':{}}
out.mkdir()
def record():
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
record()
from transformers import AutoModelForCausalLM
model=AutoModelForCausalLM.from_pretrained(source,local_files_only=True,trust_remote_code=False,
                                         dtype=torch.float32,attn_implementation='sdpa').eval()
original=list(model.parameters())
assert sum(p.numel() for p in original)==596049920
report['results']['donor']=score_prefixes(model,prefixes)
record()
print('ADJACENT_DONOR',report['results']['donor']['nats_per_token'],flush=True)
for mode in variants:
    with pair_sharing(model.model.layers,mode):
        count=sum(p.numel() for p in model.parameters())
        assert count==438763520
        result=score_prefixes(model,prefixes)
        result['registered_unique_parameters']=count
    assert [id(p) for p in model.parameters()]==[id(p) for p in original]
    report['results'][mode]=result
    record()
    print('ADJACENT_RESULT',mode,count,result['nats_per_token'],flush=True)
report['complete']=True
record()
print('ADJACENT_COMPLETE',flush=True)
