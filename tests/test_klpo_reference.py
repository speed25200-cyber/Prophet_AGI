"""Prophet's KLPO loss against the authors' reference implementation (third_party/klpo).

Same inputs, same value, same gradient: the rewrite in ``prophet/train/klpo.py`` is
checked against ``klpo.loss.klpo_token_loss`` (MC-KL route) on random tensors and on
records produced by the agent loop.
"""

import sys
from pathlib import Path

import pytest
import torch

from prophet.train.klpo import klpo_token_loss

REFERENCE = Path(__file__).resolve().parent.parent / "third_party" / "klpo"
sys.path.insert(0, str(REFERENCE))
reference = pytest.importorskip("klpo.loss")


def random_case(seed, *, b=3, t=6, v=9, m=4):
    g = torch.Generator().manual_seed(seed)
    log_p = torch.log_softmax(torch.randn(b, t, v, generator=g), -1)
    log_q = torch.log_softmax(torch.randn(b, t, v, generator=g), -1)
    tokens = torch.randint(0, v, (b, t), generator=g)
    mask = torch.rand(b, t, generator=g) < 0.6
    mask[:, 0] = False
    mask[:, 1] = True  # every row keeps at least one policy token
    draws = torch.randint(0, v, (b, t, m), generator=g)
    rewards = torch.randint(0, 2, (b,), generator=g).float()
    current = log_p.gather(-1, tokens[..., None]).squeeze(-1).clone().requires_grad_(True)
    sampler = log_q.gather(-1, tokens[..., None]).squeeze(-1)
    mc_current = log_p.gather(-1, draws).clone().requires_grad_(True)
    mc_sampler = log_q.gather(-1, draws)
    return current, sampler, rewards, mask, mc_current, mc_sampler


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("beta", [0.05, 0.1, 0.5])
def test_value_and_gradient_match_the_reference(seed, beta):
    current, sampler, rewards, mask, mc_current, mc_sampler = random_case(seed)
    ours, _ = klpo_token_loss(current, sampler, rewards, mask, mc_log_probs=mc_current, beta=beta)
    theirs, stats = reference.klpo_token_loss(
        current,
        sampler,
        rewards,
        mask,
        kl_estimator="mc",
        mc_log_probs=mc_current,
        behavior_mc_log_probs=mc_sampler,
        beta=beta,
    )
    assert torch.allclose(ours, theirs, atol=1e-6, rtol=1e-6)
    grad_ours = torch.autograd.grad(ours, (current, mc_current), retain_graph=True)
    grad_theirs = torch.autograd.grad(theirs, (current, mc_current))
    for a, b in zip(grad_ours, grad_theirs, strict=True):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-6)
    assert torch.equal(stats["policy_tokens"], mask.sum(-1))


def test_reference_contract_matches_ours_on_masking_and_length_normalisation():
    """Both sum over policy tokens and average over episodes; neither divides by length."""
    current, sampler, rewards, mask, mc_current, mc_sampler = random_case(5, b=2)
    ours, _ = klpo_token_loss(current, sampler, rewards, mask, mc_log_probs=mc_current)
    theirs, _ = reference.klpo_token_loss(
        current, sampler, rewards, mask, mc_log_probs=mc_current, behavior_mc_log_probs=mc_sampler
    )
    doubled_mask = mask.clone()
    doubled_mask[0] = True
    doubled_mask[0, 0] = False
    ours2, _ = klpo_token_loss(current, sampler, rewards, doubled_mask, mc_log_probs=mc_current)
    theirs2, _ = reference.klpo_token_loss(
        current,
        sampler,
        rewards,
        doubled_mask,
        mc_log_probs=mc_current,
        behavior_mc_log_probs=mc_sampler,
    )
    assert torch.allclose(ours, theirs) and torch.allclose(ours2, theirs2)
    assert not torch.allclose(ours, ours2), "more policy tokens must change the sum"
