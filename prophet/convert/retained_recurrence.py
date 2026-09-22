"""Experimental recurrence that retains every layer of a native Qwen2 donor.

This is an opt-in research model, not the adopted Prophet architecture. One loop
executes the original stack, with no bridge call. Later loops reuse the middle
layers and a trainable re-entry bridge. Attention caches are separate per loop:
cache memory grows with both sequence length and depth, which must be measured.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class RetainedRecurrenceConfig:
    core_start: int = 6
    core_end: int = 18
    max_loops: int = 8
    train_core: bool = False

    def validate(self, layers: int):
        for value in (self.core_start, self.core_end, self.max_loops):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError("layer boundaries and loop limit must be integers")
        if not 0 < self.core_start < self.core_end < layers or self.max_loops < 1:
            raise ValueError("nonempty prelude/core/coda and positive loop limit required")
        if not isinstance(self.train_core, bool):
            raise ValueError("train_core must be boolean")


class ReentryBridge(nn.Module):
    """Separate state/input corrections; identity initialization, no dead gate.

    Both corrections start at zero and receive gradients on the first repeated pass.
    Their separate tensors permit actual optimizer groups instead of gradient scaling
    that adaptive optimizers can cancel. The first donor pass bypasses this module.
    """

    def __init__(self, width: int, *, device=None, dtype=None):
        super().__init__()
        self.state_delta = nn.Linear(width, width, bias=False, device=device, dtype=dtype)
        self.input_delta = nn.Linear(width, width, bias=False, device=device, dtype=dtype)
        nn.init.zeros_(self.state_delta.weight)
        nn.init.zeros_(self.input_delta.weight)

    def forward(self, state: Tensor, prelude: Tensor) -> Tensor:
        return state + self.state_delta(state) + self.input_delta(prelude)


@dataclass
class RetainedCache:
    loop_k: int
    layers: list
    owner: object = field(repr=False)
    position: int = 0
    batch_size: int | None = None
    failed: bool = False

    def n_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for cache in self.layers
            for layer in cache.layers
            for tensor in (getattr(layer, "keys", None), getattr(layer, "values", None))
            if tensor is not None
        )


@dataclass
class RetainedOutput:
    logits: Tensor | None
    hidden: Tensor
    loop_k: int
    past_key_values: RetainedCache | None


class RetainedRecurrentQwen(nn.Module):
    """Own a donor with retained weights and explicit, fixed-depth cache semantics.

    The passed donor is owned by this wrapper: its gradient flags are configured here.
    With bridge-only training, one-loop outputs stay unchanged. Unfreezing the core
    permits learning but removes that post-update guarantee and requires retention
    evaluation. No model quality or deployment claim follows from identity alone.
    """

    def __init__(self, donor: nn.Module, config: RetainedRecurrenceConfig):
        super().__init__()
        from transformers import Qwen2ForCausalLM

        if not isinstance(donor, Qwen2ForCausalLM):
            raise ValueError("requires a native Qwen2ForCausalLM donor")
        config.validate(donor.config.num_hidden_layers)
        if len(donor.model.layers) != donor.config.num_hidden_layers:
            raise ValueError("donor layer count differs from its configuration")
        if any(kind != "full_attention" for kind in donor.config.layer_types):
            raise ValueError("this experiment supports full causal attention only")
        if donor.config.attention_dropout != 0:
            raise ValueError("dropout must be zero for the identity contract")
        if donor.config._attn_implementation not in ("eager", "sdpa"):
            raise ValueError("only eager and SDPA attention have been audited")
        if getattr(donor, "is_gradient_checkpointing", False):
            raise ValueError("native gradient checkpointing is not supported by this wrapper")
        self.base = donor
        self.recurrence_config = config
        self._cache_owner = object()
        weight = donor.model.embed_tokens.weight
        self.bridge = ReentryBridge(weight.shape[1], device=weight.device, dtype=weight.dtype)
        donor.requires_grad_(False)
        if config.train_core:
            for layer in donor.model.layers[config.core_start : config.core_end]:
                layer.requires_grad_(True)
        self.base.eval()

    @property
    def lm_head(self):
        return self.base.lm_head

    def train(self, mode: bool = True):
        super().train(mode)
        # Gradients remain enabled for selected core parameters; stochastic native
        # training behavior must not change the frozen one-loop reference function.
        self.base.eval()
        return self

    def get_extra_state(self):
        return {
            "format_version": 1,
            "recurrence": asdict(self.recurrence_config),
            "native_config": {
                key: value
                for key, value in self.base.config.to_dict().items()
                if key not in ("_name_or_path", "transformers_version")
            },
            "attention_implementation": self.base.config._attn_implementation,
            "parameter_dtype": str(next(self.base.parameters()).dtype),
        }

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("checkpoint recurrence contract differs")

    def new_cache(self, loop_k: int) -> RetainedCache:
        from transformers.cache_utils import DynamicCache

        self._validate_depth(loop_k)
        return RetainedCache(
            loop_k,
            [DynamicCache(config=self.base.config) for _ in range(loop_k)],
            self._cache_owner,
        )

    def _validate_depth(self, loop_k: int):
        if (
            not isinstance(loop_k, int)
            or isinstance(loop_k, bool)
            or not 1 <= loop_k <= self.recurrence_config.max_loops
        ):
            raise ValueError("loop_k must be an integer within the configured limit")

    def forward(
        self,
        input_ids: Tensor,
        *,
        loop_k: int = 1,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: RetainedCache | None = None,
        use_cache: bool = False,
        return_logits: bool = True,
    ) -> RetainedOutput:
        from transformers.masking_utils import create_causal_mask

        self._validate_depth(loop_k)
        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError("nonempty batch-by-sequence token IDs required")
        if past_key_values is not None and not use_cache:
            raise ValueError("past_key_values requires use_cache=True")
        if use_cache and torch.is_grad_enabled():
            raise ValueError("cached execution requires no_grad or inference_mode")
        cache = (
            past_key_values
            if past_key_values is not None
            else (self.new_cache(loop_k) if use_cache else None)
        )
        if cache is not None:
            if cache.owner is not self._cache_owner or cache.failed:
                raise ValueError("cache belongs to another model or a failed forward")
            if cache.loop_k != loop_k or len(cache.layers) != loop_k:
                raise ValueError("cached depth cannot change")
            if cache.batch_size is not None and cache.batch_size != input_ids.shape[0]:
                raise ValueError("cached batch size cannot change")
            if cache.layers[0].get_seq_length() != cache.position:
                raise ValueError("cache position differs from its stored keys")
        base = self.base.model
        hidden = base.embed_tokens(input_ids)
        past = cache.position if cache is not None else 0
        if position_ids is None:
            position_ids = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0) + past
        causal_mask = create_causal_mask(
            config=self.base.config,
            inputs_embeds=hidden,
            attention_mask=attention_mask,
            past_key_values=cache.layers[0] if cache is not None else None,
            position_ids=position_ids,
        )
        positions = base.rotary_emb(hidden, position_ids)
        start, end = self.recurrence_config.core_start, self.recurrence_config.core_end

        def run_layers(state, first, stop, iteration):
            for layer in base.layers[first:stop]:
                state = layer(
                    state,
                    attention_mask=causal_mask,
                    position_embeddings=positions,
                    position_ids=position_ids,
                    past_key_values=cache.layers[iteration] if cache is not None else None,
                    use_cache=use_cache,
                )
            return state

        try:
            prelude = run_layers(hidden, 0, start, 0)
            hidden = prelude
            for iteration in range(loop_k):
                if iteration:
                    hidden = self.bridge(hidden, prelude)
                hidden = run_layers(hidden, start, end, iteration)
            hidden = run_layers(hidden, end, len(base.layers), 0)
            hidden = base.norm(hidden)
            logits = self.lm_head(hidden) if return_logits else None
            if cache is not None:
                cache.position += input_ids.shape[1]
                cache.batch_size = input_ids.shape[0]
            return RetainedOutput(logits, hidden, loop_k, cache)
        except Exception:
            # A native cache may already have appended some layers. Never allow a
            # partially advanced cache to be silently reused after an exception.
            if cache is not None:
                cache.failed = True
            raise
