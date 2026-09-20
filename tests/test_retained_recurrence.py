"""Preserved donor computation, causal cache separation and effective learning."""

import copy
import io

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from prophet.convert.retained_recurrence import RetainedRecurrenceConfig, RetainedRecurrentQwen


def donor():
    torch.manual_seed(72)
    config = Qwen2Config(
        vocab_size=67,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=True,
        attention_dropout=0.0,
    )
    config._attn_implementation = "sdpa"
    return Qwen2ForCausalLM(config).float().eval()


def wrapped(*, train_core=False):
    return RetainedRecurrentQwen(donor(), RetainedRecurrenceConfig(2, 4, 4, train_core)).eval()


@pytest.mark.parametrize("padded", [False, True])
def test_first_loop_exactly_preserves_native_hidden_and_logits(padded):
    base = donor()
    ids = torch.tensor([[1, 6, 9, 3, 2], [1, 7, 4, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]) if padded else None
    with torch.no_grad():
        expected_hidden = base.model(ids, attention_mask=mask, use_cache=False).last_hidden_state
        expected = base(ids, attention_mask=mask, use_cache=False).logits
        model = RetainedRecurrentQwen(base, RetainedRecurrenceConfig(2, 4, 4))
        actual = model(ids, attention_mask=mask)
    assert torch.equal(actual.hidden, expected_hidden)
    assert torch.equal(actual.logits, expected)
    assert model.base.lm_head.weight is model.base.model.embed_tokens.weight


@pytest.mark.parametrize("loops", [1, 2, 4])
@pytest.mark.parametrize("prefix", [1, 3])
def test_full_forward_equals_prefill_and_incremental_with_separate_cache(loops, prefix):
    model = wrapped()
    ids = torch.tensor([[1, 6, 9, 3, 2, 11], [1, 7, 4, 8, 5, 12]])
    # Exercise learned re-entry, not merely its identity initialization.
    with torch.no_grad():
        model.bridge.input_delta.weight.normal_(std=0.01)
        model.bridge.state_delta.weight.normal_(std=0.01)
        expected = model(ids, loop_k=loops).logits
        output = model(ids[:, :prefix], loop_k=loops, use_cache=True)
        parts = [output.logits]
        cache = output.past_key_values
        for index in range(prefix, ids.shape[1]):
            output = model(
                ids[:, index : index + 1], loop_k=loops, use_cache=True, past_key_values=cache
            )
            parts.append(output.logits)
        actual = torch.cat(parts, dim=1)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert torch.equal(actual.argmax(-1), expected.argmax(-1))
    effective_layers = 6 + (loops - 1) * 2
    assert cache.n_bytes() == effective_layers * 2 * 2 * 6 * 2 * 8 * 4
    assert cache.position == 6
    for iteration, native_cache in enumerate(cache.layers):
        for layer_index in range(6):
            expected_length = 6 if iteration == 0 or 2 <= layer_index < 4 else 0
            assert native_cache.get_seq_length(layer_index) == expected_length
    if loops > 1:
        assert (
            cache.layers[0].layers[2].keys.data_ptr() != cache.layers[1].layers[2].keys.data_ptr()
        )


def test_left_padding_positions_and_cache_match_full_forward():
    model = wrapped()
    ids = torch.tensor([[0, 0, 1, 6, 9, 3], [0, 1, 7, 4, 8, 5]])
    mask = (ids != 0).long()
    positions = (mask.cumsum(-1) - 1).clamp_min(0)
    with torch.no_grad():
        expected = model(ids, loop_k=3, attention_mask=mask, position_ids=positions).logits
        first = model(
            ids[:, :4],
            loop_k=3,
            attention_mask=mask[:, :4],
            position_ids=positions[:, :4],
            use_cache=True,
        )
        last = model(
            ids[:, 4:],
            loop_k=3,
            attention_mask=mask,
            position_ids=positions[:, 4:],
            use_cache=True,
            past_key_values=first.past_key_values,
        )
    torch.testing.assert_close(last.logits, expected[:, 4:], atol=1e-6, rtol=1e-5)


def test_future_tokens_do_not_change_prefix_at_repeated_depth():
    model = wrapped()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    changed = torch.tensor([[1, 2, 3, 17, 19]])
    with torch.no_grad():
        left = model(ids, loop_k=4).logits
        right = model(changed, loop_k=4).logits
    assert torch.equal(left[:, :3], right[:, :3])
    assert not torch.equal(left[:, 3:], right[:, 3:])


def test_both_bridge_tensors_learn_while_frozen_one_loop_function_is_preserved():
    model = wrapped().train()
    ids = torch.tensor([[1, 9, 2, 5, 11]])
    reference = model(ids).logits.detach().clone()
    original = {name: tensor.detach().clone() for name, tensor in model.base.state_dict().items()}
    optimizer = torch.optim.AdamW(model.bridge.parameters(), lr=1e-3)
    initial_two = model(ids, loop_k=2).logits.detach().clone()
    output = model(ids, loop_k=2)
    torch.nn.functional.cross_entropy(
        output.logits[:, :-1].flatten(0, 1), ids[:, 1:].flatten()
    ).backward()
    for parameter in model.bridge.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0
    assert all(p.grad is None for p in model.base.parameters())
    optimizer.step()
    assert torch.equal(model(ids).logits, reference)
    assert not torch.equal(model(ids, loop_k=2).logits, initial_two)
    assert all(
        torch.equal(tensor, original[name]) for name, tensor in model.base.state_dict().items()
    )


def test_core_training_is_explicit_and_removes_post_update_identity_guarantee():
    model = wrapped(train_core=True)
    ids = torch.tensor([[1, 2, 3, 4]])
    before = model(ids).logits.detach().clone()
    model(ids).logits.square().sum().backward()
    assert all(p.grad is None for p in model.bridge.parameters())
    assert all(p.grad is not None for p in model.base.model.layers[2].parameters())
    assert all(p.grad is None for p in model.base.model.layers[0].parameters())
    torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01).step()
    assert not torch.equal(model(ids).logits, before)


def test_serialization_preserves_contract_and_rejects_changed_depth_partition():
    model = wrapped()
    with torch.no_grad():
        model.bridge.input_delta.weight.normal_(std=0.02)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    state = torch.load(buffer, weights_only=True)
    restored = wrapped()
    restored.load_state_dict(state)
    ids = torch.tensor([[1, 2, 3]])
    assert torch.equal(model(ids, loop_k=3).logits, restored(ids, loop_k=3).logits)
    assert restored.base.lm_head.weight is restored.base.model.embed_tokens.weight
    bad = RetainedRecurrentQwen(donor(), RetainedRecurrenceConfig(1, 4, 4))
    with pytest.raises(ValueError, match="contract"):
        bad.load_state_dict(state)


def test_cache_rejects_depth_owner_batch_and_autograd_mismatches():
    model = wrapped()
    ids = torch.tensor([[1, 2, 3]])
    with pytest.raises(ValueError, match="no_grad"):
        model(ids, use_cache=True)
    with torch.no_grad():
        cache = model(ids, loop_k=2, use_cache=True).past_key_values
        for kwargs in ({"loop_k": 1}, {"loop_k": 3}):
            with pytest.raises(ValueError, match="depth"):
                model(ids, use_cache=True, past_key_values=cache, **kwargs)
        with pytest.raises(ValueError, match="another model"):
            wrapped()(ids, loop_k=2, use_cache=True, past_key_values=cache)
        with pytest.raises(ValueError, match="batch"):
            model(ids.repeat(2, 1), loop_k=2, use_cache=True, past_key_values=cache)
    with pytest.raises(ValueError, match="requires use_cache"):
        model(ids, loop_k=2, past_key_values=cache)


def test_checkpoint_rejects_changed_native_numerical_contract():
    state = wrapped().state_dict()
    altered = wrapped()
    altered.base.config.rms_norm_eps *= 10
    with pytest.raises(ValueError, match="contract"):
        altered.load_state_dict(state)
    with pytest.raises(ValueError, match="contract"):
        wrapped().double().load_state_dict(state)


def test_failed_partially_advanced_cache_cannot_be_reused():
    model = wrapped()
    cache = model.new_cache(2)

    def fail(*args):
        raise RuntimeError("injected bridge failure")

    hook = model.bridge.register_forward_pre_hook(fail)
    with torch.no_grad(), pytest.raises(RuntimeError, match="injected"):
        model(torch.tensor([[1, 2]]), loop_k=2, use_cache=True, past_key_values=cache)
    hook.remove()
    assert cache.failed
    with torch.no_grad(), pytest.raises(ValueError, match="failed forward"):
        model(torch.tensor([[1, 2]]), loop_k=2, use_cache=True, past_key_values=cache)


def test_invalid_config_and_unsupported_donor_are_rejected():
    for config in (
        RetainedRecurrenceConfig(0, 4, 4),
        RetainedRecurrenceConfig(2, 6, 4),
        RetainedRecurrenceConfig(2, 4, 0),
        RetainedRecurrenceConfig(2, 4, True),
    ):
        with pytest.raises(ValueError):
            RetainedRecurrentQwen(donor(), config)
    bad = donor()
    bad.config.layer_types[2] = "sliding_attention"
    with pytest.raises(ValueError, match="full causal"):
        RetainedRecurrentQwen(bad, RetainedRecurrenceConfig(2, 4, 4))
    for value in (True, 0, 5, 2.5):
        with pytest.raises(ValueError, match="loop_k"):
            wrapped()(torch.tensor([[1, 2]]), loop_k=value)


def test_bridge_only_exact_optimizer_restart():
    def advance(model, optimizer, start, count):
        ids = torch.tensor([[1, 5, 9, 2, 7]])
        for step in range(start, start + count):
            optimizer.zero_grad(set_to_none=True)
            logits = model(ids, loop_k=2 + step % 2).logits
            torch.nn.functional.cross_entropy(
                logits[:, :-1].flatten(0, 1), ids[:, 1:].flatten()
            ).backward()
            optimizer.step()

    continuous, interrupted = wrapped(), wrapped()
    opt_a = torch.optim.AdamW(continuous.bridge.parameters(), lr=0.001)
    opt_b = torch.optim.AdamW(interrupted.bridge.parameters(), lr=0.001)
    advance(continuous, opt_a, 0, 4)
    advance(interrupted, opt_b, 0, 1)
    resumed = wrapped()
    resumed.load_state_dict(copy.deepcopy(interrupted.state_dict()))
    opt_c = torch.optim.AdamW(resumed.bridge.parameters(), lr=0.001)
    opt_c.load_state_dict(copy.deepcopy(opt_b.state_dict()))
    advance(resumed, opt_c, 1, 3)
    for name, tensor in continuous.state_dict().items():
        if isinstance(tensor, torch.Tensor):
            assert torch.equal(tensor, resumed.state_dict()[name])
    for index, state in opt_a.state_dict()["state"].items():
        for key, tensor in state.items():
            assert torch.equal(tensor, opt_c.state_dict()["state"][index][key])
