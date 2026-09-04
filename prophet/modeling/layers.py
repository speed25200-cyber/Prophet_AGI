"""Core layers for Prophet.

Design constraints that shape every module here:

- **Quantisation-aware from the start.** Small, over-trained models are the worst case
  for post-training quantisation, so anything that creates activation outliers is
  avoided rather than patched later: QK-norm is on by default and all norms are RMSNorm
  computed in float32.
- **Bounded state where possible.** Softmax attention keeps a cache that grows with
  context; :class:`GatedDeltaNet` keeps a fixed-size matrix state. The hybrid stack
  exists to buy exact recall from a minority of layers and constant memory from the
  majority.
- **One code path.** Training, prefill and single-token decoding share the same forward;
  passing a cache switches behaviour rather than selecting a second implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "RMSNorm",
    "RotaryEmbedding",
    "apply_rotary",
    "SwiGLU",
    "AttentionCache",
    "RecurrentState",
    "CausalSelfAttention",
    "GatedDeltaNet",
    "build_mixer",
    "HAS_FLA",
]

try:  # pragma: no cover - availability depends on the environment
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _fla_gated_delta

    HAS_FLA = True
except Exception:  # pragma: no cover
    _fla_gated_delta = None
    HAS_FLA = False


# --------------------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root-mean-square normalisation, reduced in float32.

    The reduction is where low precision actually hurts, and its cost is negligible
    next to the matmuls, so it is always done in float32 regardless of input dtype.
    """

    def __init__(self, dim: int, eps: float = 1e-5, *, elementwise_affine: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        out = x32.to(dtype)
        return out * self.weight if self.weight is not None else out

    def extra_repr(self) -> str:
        n = self.weight.numel() if self.weight is not None else 0
        return f"dim={n}, eps={self.eps}"


# --------------------------------------------------------------------------------------
# Positions
# --------------------------------------------------------------------------------------


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings with optional NTK-aware scaling.

    ``position_dims > 1`` splits the head dimension into equal sections so 2-D or 3-D
    positions (needed for image patches later) can be encoded without touching trunk
    weights. Reserving that structure now costs nothing; adding it later costs a retrain.
    """

    def __init__(
        self,
        head_dim: int,
        *,
        theta: float = 500_000.0,
        scaling: str = "none",
        scaling_factor: float = 1.0,
        position_dims: int = 1,
    ) -> None:
        super().__init__()
        if head_dim % (2 * position_dims) != 0:
            raise ValueError(
                f"head_dim={head_dim} must be divisible by 2*position_dims={2 * position_dims}"
            )
        self.head_dim = head_dim
        self.position_dims = position_dims
        self.section = head_dim // position_dims

        if scaling == "linear" and scaling_factor > 1.0:
            theta = theta * scaling_factor
        elif scaling == "yarn" and scaling_factor > 1.0:
            # NTK-by-parts: stretch the base so low frequencies span the longer context
            # while high frequencies, which carry local ordering, stay intact.
            theta = theta * scaling_factor ** (head_dim / max(head_dim - 2, 1))
        self.theta = theta

        inv_freq = 1.0 / (
            theta ** (torch.arange(0, self.section, 2, dtype=torch.float32) / self.section)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(cos, sin)`` of shape ``(batch, seq, head_dim)``.

        ``positions`` is ``(batch, seq)`` for text, or ``(batch, seq, position_dims)``
        for multi-dimensional positions.
        """
        if positions.dim() == 2:
            positions = positions.unsqueeze(-1)
        if positions.shape[-1] != self.position_dims:
            raise ValueError(
                f"expected {self.position_dims} position dims, got {positions.shape[-1]}"
            )
        freqs = positions.float().unsqueeze(-1) * self.inv_freq  # (b, s, pdims, section/2)
        freqs = freqs.flatten(-2)  # (b, s, head_dim/2)
        return torch.cat([freqs, freqs], -1).cos(), torch.cat([freqs, freqs], -1).sin()


def _rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary embeddings to ``(batch, heads, seq, head_dim)``."""
    cos = cos.unsqueeze(1).to(x.dtype)
    sin = sin.unsqueeze(1).to(x.dtype)
    return x * cos + _rotate_half(x) * sin


# --------------------------------------------------------------------------------------
# Channel mixing
# --------------------------------------------------------------------------------------


class SwiGLU(nn.Module):
    """Gated feed-forward network.

    Three matrices instead of two, so ``hidden`` is scaled by 2/3 upstream to keep the
    parameter count comparable to a plain 4x FFN (see ``prophet.budget._swiglu_hidden``).
    """

    def __init__(self, dim: int, hidden: int, *, bias: bool = False) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=bias)
        self.up_proj = nn.Linear(dim, hidden, bias=bias)
        self.down_proj = nn.Linear(hidden, dim, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# --------------------------------------------------------------------------------------
# Caches
# --------------------------------------------------------------------------------------


@dataclass
class AttentionCache:
    """Key/value cache for one softmax-attention layer.

    ``window`` bounds the retained length. Sliding-window layers therefore have a
    *constant* memory cost, which is what makes long context affordable on an 8GB phone;
    ``sink_tokens`` keeps a short always-attended prefix, without which windowed
    attention collapses at long context because the softmax has nowhere to dump
    probability mass.
    """

    keys: Tensor | None = None
    values: Tensor | None = None
    window: int | None = None
    sink_tokens: int = 0
    seen: int = 0
    """Total tokens ever written, which is what positions must be derived from — the
    buffer length is not the position once eviction has started."""

    def append(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        if self.keys is None:
            self.keys, self.values = k, v
        else:
            self.keys = torch.cat([self.keys, k], dim=2)
            self.values = torch.cat([self.values, v], dim=2)
        self.seen += k.shape[2]

        if self.window is not None:
            limit = self.window + self.sink_tokens
            length = self.keys.shape[2]
            if length > limit:
                if self.sink_tokens:
                    self.keys = torch.cat(
                        [self.keys[:, :, : self.sink_tokens], self.keys[:, :, -self.window :]],
                        dim=2,
                    )
                    self.values = torch.cat(
                        [self.values[:, :, : self.sink_tokens], self.values[:, :, -self.window :]],
                        dim=2,
                    )
                else:
                    self.keys = self.keys[:, :, -self.window :]
                    self.values = self.values[:, :, -self.window :]
        return self.keys, self.values

    def n_bytes(self) -> int:
        if self.keys is None:
            return 0
        return self.keys.numel() * self.keys.element_size() * 2


@dataclass
class RecurrentState:
    """Fixed-size state for one gated-delta layer.

    Shape ``(batch, heads, head_v, head_k)``. It does not grow with context — the
    property the whole hybrid design is built around — and it is small enough to
    serialise, which is what lets memory persist across sessions (track R03).
    """

    state: Tensor | None = None
    conv_state: Tensor | None = None
    seen: int = 0

    def n_bytes(self) -> int:
        total = 0
        for t in (self.state, self.conv_state):
            if t is not None:
                total += t.numel() * t.element_size()
        return total


# --------------------------------------------------------------------------------------
# Softmax attention
# --------------------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    """Grouped-query causal attention, optionally windowed and with attention sinks.

    ``window=None`` gives full attention (exact recall, cache linear in context);
    ``window=w`` gives a constant-memory local mixer. A Prophet stack interleaves both.
    """

    def __init__(
        self,
        dim: int,
        *,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int | None = None,
        qk_norm: bool = True,
        window: int | None = None,
        sink_tokens: int = 0,
        norm_eps: float = 1e-5,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if n_heads % n_kv_heads != 0:
            raise ValueError(f"n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads}")
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = head_dim or dim // n_heads
        self.window = window
        self.sink_tokens = sink_tokens
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(n_heads * self.head_dim, dim, bias=bias)

        # QK-norm is what keeps logits in a sane range without a soft cap, and it removes
        # the activation outliers that make small models quantise badly.
        self.q_norm = RMSNorm(self.head_dim, norm_eps) if qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, norm_eps) if qk_norm else None

    def forward(
        self,
        x: Tensor,
        *,
        cos: Tensor | None = None,
        sin: Tensor | None = None,
        cache: AttentionCache | None = None,
    ) -> Tensor:
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if cos is not None:
            q = apply_rotary(q, cos, sin)
            k = apply_rotary(k, cos, sin)

        if cache is not None:
            cache.window = self.window
            cache.sink_tokens = self.sink_tokens
            k, v = cache.append(k, v)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        kv_len = k.shape[2]
        # A single decode step attends to everything retained in the cache, so no mask is
        # needed; eviction has already enforced the window.
        if s == 1 and cache is not None:
            attn_mask, is_causal = None, False
        elif self.window is None:
            attn_mask, is_causal = None, True
        else:
            attn_mask, is_causal = self._windowed_mask(s, kv_len, x.device), False

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=self.scale
        )
        return self.o_proj(out.transpose(1, 2).reshape(b, s, -1))

    def _windowed_mask(self, q_len: int, kv_len: int, device: torch.device) -> Tensor:
        """Boolean mask (True = attend) for sliding-window attention with sinks."""
        offset = kv_len - q_len
        q_pos = torch.arange(q_len, device=device).unsqueeze(1) + offset
        k_pos = torch.arange(kv_len, device=device).unsqueeze(0)
        mask = (k_pos <= q_pos) & (q_pos - k_pos < self.window)
        if self.sink_tokens:
            mask = mask | ((k_pos < self.sink_tokens) & (k_pos <= q_pos))
        return mask.unsqueeze(0).unsqueeze(0)


# --------------------------------------------------------------------------------------
# Gated delta-rule recurrence
# --------------------------------------------------------------------------------------


class GatedDeltaNet(nn.Module):
    r"""Bounded-state sequence mixer using the gated delta rule.

    The recurrence, per head, with state :math:`S_t \in \mathbb{R}^{d_v \times d_k}`:

    .. math::
        S_t = \alpha_t\, S_{t-1} \left(I - \beta_t k_t k_t^\top\right)
              + \beta_t\, v_t k_t^\top,
        \qquad o_t = S_t q_t

    :math:`\alpha_t \in (0,1)` is a per-head forget gate and :math:`\beta_t` a write
    strength. The bracketed term is the delta rule: it *removes* the value currently
    associated with :math:`k_t` before writing the new one, which is why this family
    handles associative recall far better than plain linear attention, where writes
    simply accumulate and interfere.

    **The range of** :math:`\beta` **is load-bearing, and it is easy to get wrong.**
    With :math:`\beta \in (0,1)` every eigenvalue of :math:`\alpha(I - \beta kk^\top)`
    is strictly positive, so the transition matrix can never reflect — and a product of
    non-negative-eigenvalue transitions cannot express parity or any other problem
    requiring sign flips. Allowing :math:`\beta \in (0,2)` admits negative eigenvalues
    and recovers those problems. The cost is one multiplication; the difference is
    between chance and 0.9+ on length-generalised parity.

    Two properties matter for Prophet. The state is a fixed-size matrix, so memory is
    independent of context length. And because the update is a closed-form write rather
    than a gradient step, the same primitive serves as the persistent-memory mechanism
    in track R03 — it can be updated at inference time on a phone.

    The reference implementation is a sequential scan: correct, differentiable, and far
    too slow to train with. When ``flash-linear-attention`` is installed its fused
    chunked kernel is used instead; :func:`prophet.modeling.layers.HAS_FLA` reports which
    path is active.
    """

    def __init__(
        self,
        dim: int,
        *,
        n_heads: int = 8,
        head_dim: int = 128,
        expand: float = 2.0,
        conv_kernel: int = 4,
        norm_eps: float = 1e-5,
        bias: bool = False,
        allow_fused: bool = True,
        beta_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.beta_max = beta_max
        self.n_heads = n_heads
        self.head_k = head_dim
        self.head_v = int(head_dim * expand)
        self.conv_kernel = conv_kernel
        self.allow_fused = allow_fused

        qk_dim = n_heads * self.head_k
        v_dim = n_heads * self.head_v
        self.q_proj = nn.Linear(dim, qk_dim, bias=bias)
        self.k_proj = nn.Linear(dim, qk_dim, bias=bias)
        self.v_proj = nn.Linear(dim, v_dim, bias=bias)
        # One scalar per head: the forget gate and the write strength.
        self.a_proj = nn.Linear(dim, n_heads, bias=True)
        self.b_proj = nn.Linear(dim, n_heads, bias=True)
        self.o_proj = nn.Linear(v_dim, dim, bias=bias)
        self.o_norm = RMSNorm(self.head_v, norm_eps)

        # Short causal depthwise convolution: cheap, and it supplies the local
        # n-gram sensitivity that a pure linear recurrence is weak at.
        self.conv = nn.Conv1d(
            2 * qk_dim + v_dim,
            2 * qk_dim + v_dim,
            kernel_size=conv_kernel,
            groups=2 * qk_dim + v_dim,
            padding=0,
            bias=False,
        )
        # Initialise the forget gate near 1 (state is kept) and the write strength near
        # a moderate value: starting with an aggressively forgetting state makes the
        # layer untrainable, since gradients never reach far back.
        nn.init.constant_(self.a_proj.bias, 3.0)
        nn.init.constant_(self.b_proj.bias, 0.0)

    # -- helpers ----------------------------------------------------------------------

    def _causal_conv(self, x: Tensor, state: RecurrentState | None) -> Tensor:
        """Depthwise causal convolution over ``(batch, seq, channels)``."""
        b, s, c = x.shape
        xt = x.transpose(1, 2)  # (b, c, s)
        pad = self.conv_kernel - 1
        if state is not None and state.conv_state is not None:
            # Session files are intentionally loaded on CPU. Migrate lazily to the
            # activation device/dtype so restored state works on CUDA without making
            # the persistence layer guess where the model will run.
            restored_conv = state.conv_state.to(device=xt.device, dtype=xt.dtype)
            xt = torch.cat([restored_conv, xt], dim=2)
        else:
            xt = F.pad(xt, (pad, 0))
        if state is not None:
            state.conv_state = xt[:, :, -pad:].detach() if pad else None
        return self.conv(xt).transpose(1, 2)[:, -s:]

    def forward(
        self,
        x: Tensor,
        *,
        state: RecurrentState | None = None,
    ) -> Tensor:
        b, s, _ = x.shape
        h, dk, dv = self.n_heads, self.head_k, self.head_v

        qkv = torch.cat([self.q_proj(x), self.k_proj(x), self.v_proj(x)], dim=-1)
        qkv = F.silu(self._causal_conv(qkv, state))
        q, k, v = qkv.split([h * dk, h * dk, h * dv], dim=-1)
        q = q.view(b, s, h, dk)
        k = k.view(b, s, h, dk)
        v = v.view(b, s, h, dv)

        # L2-normalised keys keep the delta-rule update a well-conditioned projection;
        # without this the removal term can amplify rather than erase.
        k = F.normalize(k, dim=-1, eps=1e-6)
        alpha = torch.sigmoid(self.a_proj(x).float())  # (b, s, h)
        beta = self.beta_max * torch.sigmoid(self.b_proj(x).float())

        if self.allow_fused and HAS_FLA and x.is_cuda:  # pragma: no cover
            out, new_state = _fla_gated_delta(
                q=q, k=k, v=v, g=alpha.log(), beta=beta,
                initial_state=(
                    None
                    if state is None or state.state is None
                    else state.state.to(device=q.device)
                ),
                output_final_state=state is not None,
                head_first=False,
            )
        else:
            out, new_state = self._scan(q, k, v, alpha, beta, state)

        if state is not None:
            state.state = new_state
            state.seen += s

        out = self.o_norm(out)
        return self.o_proj(out.reshape(b, s, h * dv))

    def _scan(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        alpha: Tensor,
        beta: Tensor,
        state: RecurrentState | None,
    ) -> tuple[Tensor, Tensor]:
        """Reference sequential scan. Correct, differentiable, and slow by design."""
        b, s, h, dk = k.shape
        dv = v.shape[-1]
        dtype = torch.float32  # the recurrence is where precision actually matters

        S = (
            state.state.to(device=q.device, dtype=dtype)
            if state is not None and state.state is not None
            else q.new_zeros(b, h, dv, dk, dtype=dtype)
        )
        q32, k32, v32 = q.float(), k.float(), v.float()
        outputs = []
        for t in range(s):
            kt = k32[:, t].unsqueeze(-1)  # (b, h, dk, 1)
            vt = v32[:, t].unsqueeze(-1)  # (b, h, dv, 1)
            qt = q32[:, t].unsqueeze(-1)
            a = alpha[:, t].view(b, h, 1, 1)
            bt = beta[:, t].view(b, h, 1, 1)

            # Remove what k_t currently retrieves, then write v_t in its place.
            retrieved = S @ kt  # (b, h, dv, 1)
            S = a * S + bt * (vt - a * retrieved) @ kt.transpose(-1, -2)
            outputs.append((S @ qt).squeeze(-1))

        out = torch.stack(outputs, dim=1)  # (b, s, h, dv)
        return out.to(q.dtype), S.detach() if state is not None else S


# --------------------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------------------


def build_mixer(kind: str, cfg, *, layer_index: int) -> nn.Module | None:
    """Instantiate the sequence mixer for one block, per ``cfg.mixer.pattern``."""
    m = cfg.mixer
    if kind == "identity":
        return None
    if kind in ("full_attn", "swa"):
        return CausalSelfAttention(
            cfg.d_model,
            n_heads=m.n_heads,
            n_kv_heads=m.n_kv_heads,
            head_dim=cfg.head_dim,
            qk_norm=m.qk_norm,
            window=None if kind == "full_attn" else m.sliding_window,
            sink_tokens=m.attention_sink_tokens if kind == "swa" else 0,
            norm_eps=cfg.norm_eps,
        )
    if kind in ("gdn", "mamba2"):
        return GatedDeltaNet(
            cfg.d_model,
            n_heads=m.linear_heads,
            head_dim=m.linear_head_dim,
            expand=m.linear_expand,
            conv_kernel=m.conv_kernel,
            norm_eps=cfg.norm_eps,
            beta_max=m.linear_beta_max,
        )
    raise ValueError(f"unknown mixer kind: {kind!r}")
