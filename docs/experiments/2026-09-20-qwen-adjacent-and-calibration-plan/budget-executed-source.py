import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0,str(Path.cwd()))
from prophet.config import HeadsConfig
from prophet.convert.donors import get_donor
from prophet.convert.plan import prophet_config_for_donor
from prophet.budget import count_parameters

out=Path('data/qwen-covariance-budget-v1')
assert not out.exists()
out.mkdir()
config=prophet_config_for_donor(get_donor('qwen3-0.6b'),core_layers=10,loop_k=2)
config.heads=HeadsConfig(n_multi_token_predict=0,confidence_head=False)
config.mixer.pattern=['full_attn']
config.mixer.nope_layers=()
config.recurrent.core_pattern=['full_attn']
config.recurrent.inject_input_each_step=False
config.recurrent.eval_state_init='prelude'
config.recurrent.train_loop_min=config.recurrent.train_loop_max=2
config.recurrent.truncated_backprop_steps=2
config.max_seq_len=512
config.validate()
config.to_json(out/'budget-proxy-config.json')
base=count_parameters(config).total
assert base+23040==438763520,(base,base+23040)
environment={**os.environ,'PYTHONIOENCODING':'utf-8'}
commands=[('budget.txt',['-m','prophet.budget',str(out/'budget-proxy-config.json'),'--a100-hours','0.5']),
          ('search.txt',['scripts/design_search.py','--a100-hours','0.5','--top','3']),
          ('plan.txt',['-m','prophet.plan','--a100-hours','0.5'])]
for name,command in commands:
    output=subprocess.check_output([sys.executable,*command],env=environment)
    (out/name).write_bytes(output.replace(b'\r\n',b'\n'))
moment_bytes=20*(1024**2+2048**2+1024**2+3072**2)*8
report={'complete':True,'proxy_parameters':base,'extra_native_depth_norm_parameters':23040,
        'native_candidate_parameters':438763520,'moment_storage_bytes':moment_bytes,
        'moment_storage_gib':moment_bytes/2**30,'maximum_gpu_wall_seconds':1800,
        'ridge_fraction':0.01,'training_calibration_prefixes':64,'prefix_length':512,
        'scope':'Sizing proxy only: global two-loop Prophet with tied norms has the same linear parameter count as local adjacent sharing. Native scout keeps 10 extra norm sets. No model/config adoption. Budget does not measure native activation/solver/allocator peaks. Half-hour bound is a diagnostic ceiling, not a from-scratch training plan.'}
(out/'budget-scope.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(report,indent=2))
