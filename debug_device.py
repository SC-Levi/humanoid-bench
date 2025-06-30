import torch
import torch.nn as nn
from tdmpc2.common import layers
from tdmpc2.common.world_model import WorldModel
from tdmpc2.Mooretdmpc import MooreTDMPC
import yaml

# Load config
with open('tdmpc2/config.yaml', 'r') as f:
    cfg_dict = yaml.safe_load(f)

# Create a minimal config object
class Config:
    def __init__(self, cfg_dict):
        for k, v in cfg_dict.items():
            setattr(self, k, v)

cfg = Config(cfg_dict)
cfg.task = 'humanoid_h1hand-truck-v0'

print("Creating MooreTDMPC...")
agent = MooreTDMPC(cfg)

print(f"Model device: {next(agent.model.parameters()).device}")
print(f"_Qs device: {next(agent.model._Qs.parameters()).device}")
print(f"_target_Qs device: {next(agent.model._target_Qs.parameters()).device}")

# Check param_states
print(f"_target_Qs param_states[0] device:")
for name, param in list(agent.model._target_Qs.param_states[0].items())[:3]:
    print(f"  {name}: {param.device}")

print("Done") 