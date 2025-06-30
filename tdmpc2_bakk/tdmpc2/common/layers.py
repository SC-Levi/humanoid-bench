import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tensordict.nn import make_functional
from tensordict import TensorDict
from copy import deepcopy
from tdmpc2.common.mixture_layers import InputLayer, ParallelLayer, OrthogonalLayer1D

class MoEBlock(nn.Module):
	def __init__(self, cfg,
					in_dim, gate_dim, hidden_dims, out_dim,
					n_experts, use_orthogonal=False, act=None,
					# ---------- τ & 正则 ----------
					tau_init=1.8,
					tau_min=0.5,               # ③ 降低下限，允许更深的专精
					tau_max=2.0,
					beta=0.02,                 # ③ 降低反馈速率，防止过快触底
					H_target=None,             # 目标熵
					lb_alpha=3e-2,             # ⑤ 提高正则权重，从1e-2→3e-2
					# ---------- 训练调度 ----------
					total_steps=200_000,
					freeze_frac=0.05,          # 冻结 τ 的前百分比
					):
		super().__init__()
		self.n_experts = n_experts

		# ------------ 温度调度 ------------
		self.tau       = tau_init
		self.tau_min   = tau_min
		self.tau_max   = tau_max
		self.beta      = beta
		self.H_target  = H_target or 0.75 * math.log(n_experts)
		self.lb_alpha  = lb_alpha
		self.total_steps   = total_steps
		# ④ 防止freeze期过长，最少保证5000步
		self.freeze_steps  = max(5_000, int(freeze_frac * total_steps))
		self.register_buffer("global_step", torch.tensor(0, dtype=torch.long))

		# ---------------- Gate ----------------
		# ⑥ 扩展gate输入维度，为上下文信息预留空间
		self.gate = nn.Linear(gate_dim + 1, n_experts, bias=False)  # +1 for context
		with torch.no_grad():
			self.gate.weight += 1e-3 * torch.randn_like(self.gate.weight)

		# ① 累积式辅助损失，使用buffer避免梯度追踪
		self.gate_history = []
		self.register_buffer("_aux_loss", torch.tensor(0.0))

		# 2) 专家 unit：只是做特征提取，不用 SimNorm
		unit = []
		last = in_dim
		for h in hidden_dims:
			# 前几层都用 NormedLinear + 默认激活（Mish）
			unit.append(NormedLinear(last, h))
			last = h
		self.unit_model = nn.Sequential(*unit)

		# 3) 干-支并行
		trunk = [
			InputLayer(n_models=n_experts),
			ParallelLayer(self.unit_model)
		]
		if use_orthogonal:
			trunk.append(OrthogonalLayer1D())
		self.trunk = nn.Sequential(*trunk)

		if act is None:
			self.head = NormedLinear(hidden_dims[-1], out_dim)
		else:
			self.head = NormedLinear(hidden_dims[-1], out_dim, act=act)

	# ② 提供清晰的辅助损失接口
	@property
	def aux_loss(self):
		"""获取当前累积的辅助损失"""
		return self._aux_loss

	def zero_aux_loss(self):
		"""重置辅助损失累积器"""
		self._aux_loss.zero_()

	# ---------- 内部：更新温度 ----------
	def _update_tau(self, entropy: torch.Tensor):
		"""熵反馈 + 上下界裁剪"""
		if self.global_step < self.freeze_steps:
			return  # 冻结阶段：τ 保持初值
		# 自适应调整
		self.tau += self.beta * math.tanh(self.H_target - entropy.item())
		self.tau = float(torch.clamp(
			torch.tensor(self.tau), self.tau_min, self.tau_max))

	def forward(self, z, a, task_emb=None):
		# ---- 1. 区分时序 vs 单步 ----
		is_seq = (z.ndim == 3)
		if is_seq:
			T, B, _ = z.shape
			# 序列模式下，如果 task_emb 是 [B, Dt]，先扩成 [T, B, Dt]
			if task_emb is not None and task_emb.ndim == 2:
				task_emb = task_emb.unsqueeze(0).expand(T, B, -1)
			# flatten 序列和 batch
			z = z.reshape(T * B, -1)
			a = a.reshape(T * B, -1)
			if task_emb is not None:
				task_emb = task_emb.reshape(T * B, -1)
		else:
			# 单步模式下，如果 task_emb 是 [1, Dt]，扩成 [B, Dt]
			if task_emb is not None and task_emb.ndim == 2 and task_emb.shape[0] == 1:
				task_emb = task_emb.repeat(z.shape[0], 1)

		# ---- 2. 干-支并行 + 聚合 ----
		x = torch.cat([z, a] + ([task_emb] if task_emb is not None else []), dim=-1)
		feats = self.trunk(x)             # [K, N, H]
		feats = feats.permute(1, 0, 2)    # [N, K, H]

		# ⑥ 添加上下文信息到gate输入（使用当前τ作为phase indicator）
		gate_base = task_emb if task_emb is not None else torch.cat([z, a], dim=-1)
		gate_ctx = torch.full_like(z[:, :1], self.tau)  # shape [N,1]
		gate_in = torch.cat([gate_base, gate_ctx], dim=-1)
		
		logits = self.gate(gate_in) / self.tau
		w = F.softmax(logits, dim=-1)           # [N, K]

		# ⑤ 使用KL散度的负载均衡正则，更线性、更强惩罚
		probs_mean = w.mean(dim=0)                  # [K]
		uniform_dist = torch.full_like(probs_mean, 1.0 / self.n_experts)
		kl_loss = F.kl_div(
			(probs_mean + 1e-9).log(), 
			uniform_dist, 
			reduction='sum'
		)
		lb_loss = self.lb_alpha * kl_loss
		
		# ① 累加而非覆盖辅助损失
		self._aux_loss += lb_loss.detach()  # detach防止梯度爆炸
		
		# 记录 gate 权重
		if is_seq:
			# 序列输入，reshape 成 [T,B,K] 再对 batch 取平均
			w_seq = w.view(T, B, -1).mean(dim=1)  # [T, K]
			self.gate_history.append(w_seq.detach().cpu())
		else:
			# 单步输入，w 是 [B,K]，对 batch 取平均
			self.gate_history.append(w.mean(dim=0, keepdim=True).detach().cpu())  # [1,K]

		# -------- τ 调度 ----------
		if self.training:
			entropy = (-w * w.clamp_min(1e-9).log()).sum(-1).mean()
			self._last_entropy = entropy.detach()
			self._update_tau(entropy)
			self.global_step += 1
			
		w = w.unsqueeze(-1)  # [N, K, 1]

		# 聚合
		agg = (w * feats).sum(dim=1)      # [N, H]

		# head
		out = self.head(agg)              # [N, out_dim]

		# ---- 3. 如果序列展开过，修回形状 ----
		if is_seq:
			out = out.view(T, B, -1)

		return out
	
class Ensemble(nn.Module):
	"""
	Vectorized ensemble of modules.
	"""

	def __init__(self, modules, **kwargs):
		super().__init__()
		self.modules_list = nn.ModuleList(modules)
		self.module = modules[0]  # Keep for compatibility
		self._n = len(modules)
		self._repr = repr(self.module)
		
		# Create param_states for compatibility (but we won't use them for forward)
		self.param_states = []
		for module in modules:
			param_dict = dict(module.named_parameters())
			self.param_states.append(param_dict)
		
		# Create a params attribute for compatibility
		from tensordict import TensorDict
		self.params = self._create_params_tensordict()

	def _create_params_tensordict(self):
		"""Create a TensorDict structure that mimics the old params interface"""
		if not self.param_states:
			from tensordict import TensorDict
			return TensorDict({}, [])
		
		# Get the structure from the first module
		first_params = self.param_states[0]
		params_dict = {}
		
		# Create nested structure for each parameter
		for param_name, param_tensor in first_params.items():
			# Split name by dots to create nested structure
			parts = param_name.split('.')
			current_dict = params_dict
			
			for i, part in enumerate(parts[:-1]):
				if part not in current_dict:
					current_dict[part] = {}
				current_dict = current_dict[part]
			
			# Create tensor stack for the final parameter
			param_stack = torch.stack([state[param_name] for state in self.param_states])
			current_dict[parts[-1]] = param_stack
			
		# Convert to TensorDict
		from tensordict import TensorDict
		return TensorDict(params_dict, [len(self.param_states)])

	def __len__(self):
		return self._n

	def forward(self, *args, **kwargs):
		# Directly use the stored modules instead of functional_call
		results = []
		for module in self.modules_list:
			result = module(*args, **kwargs)
			results.append(result)
		return torch.stack(results)

	def __repr__(self):
		return f'Vectorized {len(self)}x ' + self._repr

	def to(self, *args, **kwargs):
		"""
		Move the Ensemble to device/dtype, ensuring param_states are also moved.
		"""
		# First move the parent module and template module
		super().to(*args, **kwargs)
		self.module = self.module.to(*args, **kwargs)
		
		# Then move all parameters in param_states
		device = None
		dtype = None
		for arg in args:
			if isinstance(arg, torch.device):
				device = arg
			elif isinstance(arg, torch.dtype):
				dtype = arg
		
		if 'device' in kwargs:
			device = kwargs['device']
		if 'dtype' in kwargs:
			dtype = kwargs['dtype']
			
		if device is not None or dtype is not None:
			new_param_states = []
			for param_dict in self.param_states:
				new_param_dict = {}
				for name, param in param_dict.items():
					new_param = param.to(*args, **kwargs)
					new_param_dict[name] = new_param
				new_param_states.append(new_param_dict)
			self.param_states = new_param_states
			
			# Recreate params TensorDict with moved parameters
			from tensordict import TensorDict
			self.params = self._create_params_tensordict()
			
		return self


class ShiftAug(nn.Module):
	"""
	Random shift image augmentation.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	def __init__(self, pad=3):
		super().__init__()
		self.pad = pad
		self.padding = tuple([self.pad] * 4)

	def forward(self, x):
		x = x.float()
		n, _, h, w = x.size()
		assert h == w
		x = F.pad(x, self.padding, 'replicate')
		eps = 1.0 / (h + 2 * self.pad)
		arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
		arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
		base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
		base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
		shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
		shift *= 2.0 / (h + 2 * self.pad)
		grid = base_grid + shift
		return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)


class PixelPreprocess(nn.Module):
	"""
	Normalizes pixel observations to [-0.5, 0.5].
	"""

	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x.div(255.).sub(0.5)


class SimNorm(nn.Module):
	"""
	Simplicial normalization.
	Adapted from https://arxiv.org/abs/2204.00616.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.dim = cfg.simnorm_dim

	def forward(self, x):
		shp = x.shape
		x = x.view(*shp[:-1], -1, self.dim)
		x = F.softmax(x, dim=-1)
		return x.view(*shp)

	def __repr__(self):
		return f"SimNorm(dim={self.dim})"


class NormedLinear(nn.Linear):
	"""
	Linear layer with LayerNorm, activation, and optionally dropout.
	"""

	def __init__(self, *args, dropout=0., act=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.ln = nn.LayerNorm(self.out_features)
		if act is None:
			act = nn.Mish(inplace=False)
		self.act = act
		self.dropout = nn.Dropout(dropout, inplace=False) if dropout else None

	def forward(self, x):
		x = super().forward(x)
		if self.dropout:
			x = self.dropout(x)
		return self.act(self.ln(x))

	def __repr__(self):
		repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
		return f"NormedLinear(in_features={self.in_features}, "\
			f"out_features={self.out_features}, "\
			f"bias={self.bias is not None}{repr_dropout}, "\
			f"act={self.act.__class__.__name__})"


def mlp(in_dim, mlp_dims, out_dim, act=None, dropout=0.):
	"""
	Basic building block of TD-MPC2.
	MLP with LayerNorm, Mish activations, and optionally dropout.
	"""
	if isinstance(mlp_dims, int):
		mlp_dims = [mlp_dims]
	dims = [in_dim] + mlp_dims + [out_dim]
	mlp = nn.ModuleList()
	for i in range(len(dims) - 2):
		mlp.append(NormedLinear(dims[i], dims[i+1], dropout=dropout*(i==0)))
	mlp.append(NormedLinear(dims[-2], dims[-1], act=act) if act else nn.Linear(dims[-2], dims[-1]))
	return nn.Sequential(*mlp)


def conv(in_shape, num_channels, act=None):
	"""
	Basic convolutional encoder for TD-MPC2 with raw image observations.
	4 layers of convolution with ReLU activations, followed by a linear layer.
	"""
	assert in_shape[-1] == 64 # assumes rgb observations to be 64x64
	layers = [
		ShiftAug(), PixelPreprocess(),
		nn.Conv2d(in_shape[0], num_channels, 7, stride=2), nn.ReLU(inplace=False),
		nn.Conv2d(num_channels, num_channels, 5, stride=2), nn.ReLU(inplace=False),
		nn.Conv2d(num_channels, num_channels, 3, stride=2), nn.ReLU(inplace=False),
		nn.Conv2d(num_channels, num_channels, 3, stride=1), nn.Flatten()]
	if act:
		layers.append(act)
	return nn.Sequential(*layers)


def enc(cfg, out={}):
	"""
	Returns a dictionary of encoders for each observation in the dict.
	"""
	for k in cfg.obs_shape.keys():
		if k == 'state':
			out[k] = mlp(cfg.obs_shape[k][0] + cfg.task_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, act=SimNorm(cfg))
		elif k == 'rgb':
			out[k] = conv(cfg.obs_shape[k], cfg.num_channels, act=SimNorm(cfg))
		else:
			raise NotImplementedError(f"Encoder for observation type {k} not implemented.")
	return nn.ModuleDict(out)


def api_model_conversion(target_state_dict, source_state_dict):
	"""
	Converts a checkpoint from our old API to the new torch.compile compatible API.
	"""
	# check whether checkpoint is already in the new format
	if "_detach_Qs_params.0.weight" in source_state_dict:
		return source_state_dict

	name_map = ['weight', 'bias', 'ln.weight', 'ln.bias']
	new_state_dict = dict()

	# rename keys
	for key, val in list(source_state_dict.items()):
		if key.startswith('_Qs.'):
			num = key[len('_Qs.params.'):]
			new_key = str(int(num) // 4) + "." + name_map[int(num) % 4]
			new_total_key = "_Qs.params." + new_key
			del source_state_dict[key]
			new_state_dict[new_total_key] = val
			new_total_key = "_detach_Qs_params." + new_key
			new_state_dict[new_total_key] = val
		elif key.startswith('_target_Qs.'):
			num = key[len('_target_Qs.params.'):]
			new_key = str(int(num) // 4) + "." + name_map[int(num) % 4]
			new_total_key = "_target_Qs_params." + new_key
			del source_state_dict[key]
			new_state_dict[new_total_key] = val

	# add batch_size and device from target_state_dict to new_state_dict
	for prefix in ('_Qs.', '_detach_Qs_', '_target_Qs_'):
		for key in ('__batch_size', '__device'):
			new_key = prefix + 'params.' + key
			new_state_dict[new_key] = target_state_dict[new_key]

	# check that every key in new_state_dict is in target_state_dict
	for key in new_state_dict.keys():
		assert key in target_state_dict, f"key {key} not in target_state_dict"
	# check that all Qs keys in target_state_dict are in new_state_dict
	for key in target_state_dict.keys():
		if 'Qs' in key:
			assert key in new_state_dict, f"key {key} not in new_state_dict"
	# check that source_state_dict contains no Qs keys
	for key in source_state_dict.keys():
		assert 'Qs' not in key, f"key {key} contains 'Qs'"

	# copy log_std_min and log_std_max from target_state_dict to new_state_dict
	new_state_dict['log_std_min'] = target_state_dict['log_std_min']
	new_state_dict['log_std_dif'] = target_state_dict['log_std_dif']
	new_state_dict['_action_masks'] = target_state_dict['_action_masks']

	# copy new_state_dict to source_state_dict
	source_state_dict.update(new_state_dict)

	return source_state_dict
