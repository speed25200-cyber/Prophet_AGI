"""Numerical and retained-workspace checks for the optional first-order loss."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from prophet.modeling.model import ProphetOutput
from prophet.train.chunked_loss import shifted_token_losses, token_losses
from prophet.train.loss import compute_loss


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("chunk", [1, 7, 100])
@pytest.mark.parametrize("objective", ["ce", "z", "both"])
def test_values_and_weighted_gradients_match_autograd(dtype, chunk, objective):
    torch.manual_seed(27)
    # Transposition also exercises a non-contiguous input and the batch boundaries.
    x = torch.randn(3, 2, 19).to(dtype).transpose(0, 1).requires_grad_()
    ref = x.detach().clone().requires_grad_()
    target = torch.randint(0, 19, (2, 3))
    target[0, 1] = -100
    ce_weight, z_weight = torch.randn(2, 3), torch.randn(2, 3) * 0.03
    ce, z = token_losses(x, target, chunk)
    expected_ce = F.cross_entropy(ref.reshape(-1, 19).float(), target.flatten(),
                                  reduction="none").view_as(target)
    expected_z = ref.float().logsumexp(-1).square()
    torch.testing.assert_close(ce, expected_ce)
    torch.testing.assert_close(z, expected_z)
    actual = sum((a * w).sum() for a, w, keep in
                 [(ce, ce_weight, objective != "z"), (z, z_weight, objective != "ce")] if keep)
    expected = sum((a * w).sum() for a, w, keep in
                   [(expected_ce, ce_weight, objective != "z"),
                    (expected_z, z_weight, objective != "ce")] if keep)
    actual.backward()
    expected.backward()
    # The fused gradient rounds once to bf16; separate reference branches round twice.
    torch.testing.assert_close(x.grad, ref.grad, rtol=0.012 if dtype == torch.bfloat16 else 2e-5,
                               atol=0.004 if dtype == torch.bfloat16 else 2e-7)


def test_large_common_offset_ce_and_ignored_targets():
    torch.manual_seed(1)
    x = (torch.randn(2, 4, 13) + 10000).requires_grad_()
    ref = x.detach().clone().requires_grad_()
    target = torch.full((2, 4), -100, dtype=torch.long)
    target[1, 2] = 3
    ce, z = token_losses(x, target, 3)
    ce.sum().backward()
    F.cross_entropy(ref.reshape(-1, 13), target.flatten(), reduction="sum").backward()
    torch.testing.assert_close(x.grad, ref.grad, rtol=1e-6, atol=1e-7)
    assert torch.isfinite(z).all()
    assert ce[target == -100].eq(0).all()
    assert x.grad[target == -100].eq(0).all()


@pytest.mark.parametrize("all_ignored", [False, True])
def test_mtp_ponder_and_jumped_lm_gradients(all_ignored):
    def run(chunk):
        torch.manual_seed(84)
        h = [torch.randn(2, 5, 7, requires_grad=True) for _ in range(3)]
        projection = torch.randn(7, 17, requires_grad=True)
        halt = torch.randn(2, 5, 3, requires_grad=True)
        target = torch.randint(0, 17, (2, 5))
        target[0, 2] = -100
        if all_ignored:
            target.fill_(-100)
        out = ProphetOutput(logits=h[-1] @ projection, hidden=h[-1], loop_k=3,
                            mtp_logits=[h[0] @ projection, h[1] @ projection],
                            halt_probs=halt.softmax(-1), hidden_per_step=h)
        action = SimpleNamespace(jumped=torch.tensor([[0, 1, 0, 0, 1], [0, 0, 1, 0, 0]]))
        loss = compute_loss(out, target, loss_chunk_tokens=chunk, ponder_weight=0.07,
                            project=lambda v: v @ projection, action_targets=action,
                            jumped_lm_weight=0.2)
        loss.total.backward()
        return loss, [v.grad for v in [*h, projection, halt]]
    expected, expected_grads = run(None)
    actual, grads = run(3)
    for name in ("total", "lm", "mtp", "z", "ponder"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name))
    for grad, expected_grad in zip(grads, expected_grads, strict=True):
        torch.testing.assert_close(grad, expected_grad, atol=2e-7, rtol=2e-5)


def test_retains_only_original_logits_and_linear_token_metadata():
    x = torch.randn(2, 9, 101, dtype=torch.bfloat16, requires_grad=True)
    targets = torch.randint(0, 101, (2, 9))
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
        ce, z = token_losses(x, targets, 4)
    large = [t for t in saved if t.numel() > targets.numel()]
    assert len(large) == 1
    assert large[0].data_ptr() == x.data_ptr()
    assert sum(t.numel() for t in saved if t.data_ptr() != x.data_ptr()) <= 2 * targets.numel()
    (ce.mean() + 1e-4 * z.mean()).backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("offset", [1, 2, 5, 8])
def test_shift_never_crosses_batch_boundaries(offset):
    torch.manual_seed(9)
    x = torch.randn(2, 5, 11, requires_grad=True)
    target = torch.randint(0, 11, (2, 5))
    ce, z = shifted_token_losses(x, target, offset, 3)
    assert ce.shape == (2, max(0, 5 - offset))
    if offset < 5:
        expected = F.cross_entropy(x[:, :-offset].reshape(-1, 11),
                                  target[:, offset:].reshape(-1), reduction="none")
        torch.testing.assert_close(ce.flatten(), expected)
    torch.testing.assert_close(z, x.logsumexp(-1).square())


@pytest.mark.parametrize("chunk", [0, -1, True, 1.5])
def test_invalid_chunk_rejected(chunk):
    with pytest.raises(ValueError, match="positive integer"):
        token_losses(torch.zeros(1, 2, 7), torch.zeros(1, 2, dtype=torch.long), chunk)


@pytest.mark.parametrize("offset", [0, -1, True])
def test_invalid_offset_rejected(offset):
    with pytest.raises(ValueError, match="offset"):
        shifted_token_losses(torch.zeros(1, 2, 7), torch.zeros(1, 2, dtype=torch.long), offset, 2)
