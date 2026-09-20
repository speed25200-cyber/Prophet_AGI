"""Bounded CPU diagnostic of depth-specific low-rank residuals; no model adoption."""
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path.cwd()))
import torch
from safetensors import safe_open
from prophet.budget import count_parameters
from prophet.config import ProphetConfig
from scripts.recover_qwen import load_source, REVISION, WEIGHTS_SHA256

torch.set_num_threads(2)
torch.manual_seed(0)
out=Path('data/donor-residual-spectrum-v1')
assert not out.exists()
out.mkdir()
source=Path('data/donor-qwen3-0.6b/source')
source_config,_=load_source(source)
audit=json.loads(Path('docs/experiments/2026-09-20-qwen-recovery-attention-kl-final/attention-kl-final-audit.json').read_bytes())
config=ProphetConfig.from_dict(audit['config'])
base_count=count_parameters(config).total
assert base_count == 360087809
(out/'base-config.json').write_text(json.dumps(audit['config'],indent=2)+'\n',encoding='utf-8',newline='\n')
commands=[('base-budget.txt',['-m','prophet.budget',str(out/'base-config.json'),'--a100-hours','6']),
          ('design-search.txt',['scripts/design_search.py','--a100-hours','6','--top','3']),
          ('plan.txt',['-m','prophet.plan','--a100-hours','6'])]
for name,args in commands:
    with (out/name).open('x',encoding='utf-8') as stream:
        subprocess.run([sys.executable,*args],stdout=stream,stderr=subprocess.STDOUT,check=True)

ranks=[16,32,64,128,256,512,1024]
projections=['self_attn.q_proj.weight','self_attn.k_proj.weight','self_attn.v_proj.weight',
             'self_attn.o_proj.weight','mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight']
norms=['input_layernorm.weight','post_attention_layernorm.weight','self_attn.q_norm.weight','self_attn.k_norm.weight']
rows=[]; total_norm_parameters=0
started=time.perf_counter()
with safe_open(source/'model.safetensors',framework='pt',device='cpu') as sf:
    for layer in range(4,24):
        total_norm_parameters += sum(sf.get_tensor(f'model.layers.{layer}.{key}').numel() for key in norms)
    for key in projections:
        weights={layer:sf.get_tensor(f'model.layers.{layer}.{key}').float() for layer in range(4,24)}
        for slot in range(4):
            cyclic=list(range(4+slot,24,4))
            contiguous=list(range(4+5*slot,9+5*slot))
            cyclic_mean=torch.stack([weights[i] for i in cyclic]).mean(0)
            contiguous_mean=torch.stack([weights[i] for i in contiguous]).mean(0)
            for layer in cyclic:
                residual=weights[layer]-cyclic_mean
                s=torch.linalg.svdvals(residual)
                energy=float(residual.double().square().sum())
                energies=s.double().square()
                spectral_total=float(energies.sum())
                assert math.isclose(spectral_total,energy,rel_tol=2e-6)
                row={'key':key,'donor_layer':layer,'shared_slot':slot,'shape':list(residual.shape),
                     'cyclic_group':cyclic,'current_contiguous_group':contiguous,
                     'cyclic_mean_residual_squared_frobenius':energy,
                     'contiguous_mean_residual_squared_frobenius':float((weights[layer]-contiguous_mean).double().square().sum()),
                     'singular_energy_relative_error':abs(spectral_total-energy)/energy,
                     'ranks':{}}
                for rank in ranks:
                    assert rank <= len(s)
                    captured=float(energies[:rank].sum())/spectral_total
                    row['ranks'][str(rank)]={'residual_energy_retained':captured,
                                            'factor_parameters':rank*sum(residual.shape)}
                rows.append(row)
                print('SPECTRUM',len(rows),140,key,layer,round(row['ranks']['64']['residual_energy_retained'],4),flush=True)
        del weights
assert len(rows)==140 and total_norm_parameters==46080
total_energy=sum(row['cyclic_mean_residual_squared_frobenius'] for row in rows)
summary={}
for rank in ranks:
    added=sum(row['ranks'][str(rank)]['factor_parameters'] for row in rows)+total_norm_parameters
    retained=sum(row['cyclic_mean_residual_squared_frobenius']*row['ranks'][str(rank)]['residual_energy_retained'] for row in rows)/total_energy
    summary[str(rank)]={'added_parameters_including_norm_residuals':added,
                        'total_parameters':base_count+added,
                        'fraction_donor_parameters_saved':1-(base_count+added)/596049920,
                        'weighted_matrix_residual_energy_retained':retained,
                        'extra_fp32_weight_bytes':4*added,
                        'extra_fp32_weights_gradients_two_moments_bytes':16*added,
                        'extra_ideal_int4_weight_bytes':added/2,
                        'extra_linear_forward_flops_per_token':2*(added-total_norm_parameters),
                        'within_500m_ablation_size':base_count+added<=500000000}
report={'protocol':'qwen-depth-residual-spectrum-v1','complete':True,'device':'cpu','torch':str(torch.__version__),
        'threads':2,'svd_dtype':'float32','energy_reductions_dtype':'float64','donor_revision':REVISION,
        'donor_weights_sha256':WEIGHTS_SHA256,'base_unique_parameters':base_count,'donor_parameters':596049920,
        'plan_horizon_hours':6,'plan_scope':'Hypothetical short diagnostic horizon, not a purchase or allocation; budget tools describe the existing base architecture, not an implemented adapter model. Memory extensions below are explicit arithmetic and exclude activations/workspace/scales.',
        'mapping':'Original serial donor layer 4+4*iteration+slot; each of four base slots averages its five cyclic donor layers. No reinjection in a hypothetical serial reconstruction. Current contiguous-mean error is included as a control; no model forward or training performed.',
        'norm_residual_parameters':total_norm_parameters,'ranks':summary,'matrices':rows,
        'seconds':time.perf_counter()-started,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'Exact singular-value spectra of every middle-layer linear residual. Best rank-r matrix approximation energy is not language quality, a trained result, adaptive-depth support, deployment memory or architecture adoption. No GPU training and no benchmark examples used.'}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
print('SPECTRUM_COMPLETE',json.dumps(summary),flush=True)
