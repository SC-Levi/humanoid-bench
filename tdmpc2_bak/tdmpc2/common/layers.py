import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch import combine_state_for_ensemble


class Ensemble(nn.Module):
    """
    Vectorized ensemble of modules.
    """

    def __init__(self, modules, **kwargs):
        super().__init__()
        modules = nn.ModuleList(modules)
        fn, params, _ = combine_state_for_ensemble(modules)
        self.vmap = torch.vmap(
            fn, in_dims=(0, 0, None), randomness="different", **kwargs
        )
        self.params = nn.ParameterList([nn.Parameter(p) for p in params])
        self._repr = str(modules)

    def forward(self, *args, **kwargs):
        return self.vmap([p for p in self.params], (), *args, **kwargs)

    def __repr__(self):
        return "Vectorized " + self._repr


class ShiftAug(nn.Module):
    """
    Random shift image augmentation.
    Adapted from https://github.com/facebookresearch/drqv2
    """

    def __init__(self, pad=3):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = x.float()
        n, _, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)
        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class PixelPreprocess(nn.Module):
    """
    Normalizes pixel observations to [-0.5, 0.5].
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.div_(255.0).sub_(0.5)


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

    def __init__(self, *args, dropout=0.0, act=nn.Mish(inplace=True), **kwargs):
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(self.out_features)
        self.act = act
        self.dropout = nn.Dropout(dropout, inplace=True) if dropout else None

    def forward(self, x):
        x = super().forward(x)
        if self.dropout:
            x = self.dropout(x)
        return self.act(self.ln(x))

    def __repr__(self):
        repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
        return (
            f"NormedLinear(in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}{repr_dropout}, "
            f"act={self.act.__class__.__name__})"
        )


def mlp(in_dim, mlp_dims, out_dim, act=None, dropout=0.0):
    """
    Basic building block of TD-MPC2.
    MLP with LayerNorm, Mish activations, and optionally dropout.
    """
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + mlp_dims + [out_dim]
    mlp = nn.ModuleList()
    for i in range(len(dims) - 2):
        mlp.append(NormedLinear(dims[i], dims[i + 1], dropout=dropout * (i == 0)))
    mlp.append(
        NormedLinear(dims[-2], dims[-1], act=act)
        if act
        else nn.Linear(dims[-2], dims[-1])
    )
    return nn.Sequential(*mlp)


def conv(in_shape, num_channels, act=None):
    """
    Basic convolutional encoder for TD-MPC2 with raw image observations.
    4 layers of convolution with ReLU activations, followed by a linear layer.
    """
    assert in_shape[-1] == 64  # assumes rgb observations to be 64x64
    layers = [
        ShiftAug(),
        PixelPreprocess(),
        nn.Conv2d(in_shape[0], num_channels, 7, stride=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(num_channels, num_channels, 5, stride=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(num_channels, num_channels, 3, stride=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(num_channels, num_channels, 3, stride=1),
        nn.Flatten(),
    ]
    if act:
        layers.append(act)
    return nn.Sequential(*layers)


def enc(cfg, out={}):
    """
    Returns a dictionary of encoders for each observation in the dict.
    """
    for k in cfg.obs_shape.keys():
        if k == "state":
            out[k] = mlp(
                cfg.obs_shape[k][0] + cfg.task_dim,
                max(cfg.num_enc_layers - 1, 1) * [cfg.enc_dim],
                cfg.latent_dim,
                act=SimNorm(cfg),
            )
        elif k == "rgb":
            out[k] = conv(cfg.obs_shape[k], cfg.num_channels, act=SimNorm(cfg))
        else:
            raise NotImplementedError(
                f"Encoder for observation type {k} not implemented."
            )
    return nn.ModuleDict(out)


from tdmpc2.common.mixture_layers import InputLayer, ParallelLayer, OrthogonalLayer1D

class MoEBlock(nn.Module):
	def __init__(self, cfg,
					in_dim, gate_dim, hidden_dims, out_dim,
					n_experts, use_orthogonal=False, act=None):
		super().__init__()
		self.n_experts = n_experts

		# 1) 门控网络
		self.gate = nn.Linear(gate_dim, n_experts, bias=False)
		# 用来存历史权重
		self.gate_history = []

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

		# gate
		gate_in = task_emb if task_emb is not None else torch.cat([z, a], dim=-1)
		logits = self.gate(gate_in)             # [N, K]
		w = F.softmax(logits, dim=-1)           # [N, K]
		
		# 记录 gate 权重
		if is_seq:
            # 序列输入，reshape 成 [T,B,K] 再对 batch 取平均
			w_seq = w.view(T, B, -1).mean(dim=1)  # [T, K]
			self.gate_history.append(w_seq.detach())
		else:
            # 单步输入，w 是 [B,K]，对 batch 取平均
			self.gate_history.append(w.mean(dim=0, keepdim=True).detach())  # [1,K]
			
		w = w.unsqueeze(-1)  # [N, K, 1]

		# 聚合
		agg = (w * feats).sum(dim=1)      # [N, H]

		# head
		out = self.head(agg)              # [N, out_dim]

		# ---- 3. 如果序列展开过，修回形状 ----
		if is_seq:
			out = out.view(T, B, -1)

		return out