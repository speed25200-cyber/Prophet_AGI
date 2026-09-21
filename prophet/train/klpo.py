"""KLPO token regression with Monte Carlo KL, in Prophet's terms (docs/research/A5_klpo.md).

The thesis (Zhang et al., 2026): regularised local policy improvement around the
*sampler* ``q`` -- the policy that actually produced the episode, however stale -- has a
Gibbs optimum whose optimality condition is a regression target on the trainer-to-sampler
log-ratio. With a terminal reward ``R`` only, every generated token ``a_u`` gets its own
coefficient and a sampler-centred score::

    ell_u = log p(a_u) - log q(a_u)
    h_u   = R - beta * ell_u                       (detached)
    z_u   = log p(a_u) - mean_j log p(v_j),  v_j ~ q iid   (M >= 1 auxiliary token draws)
    loss  = - mean_episodes sum_u h_u * z_u

The autodiff gradient of ``loss`` equals, in expectation over the draws, the gradient of
the KL-regularised regression objective. No critic, no value normaliser, no group of
responses per prompt, no ratio clipping. Tokens the model did not sample itself (prompt,
observations, control ids, spliced values, greedy tokens) are excluded: for a point-mass
sampler the score correction is identically zero and there is nothing to learn from.

What is Prophet-specific here: the sampler distribution is the one the agent loop
actually drew from (grammar-masked, tempered), recorded at generation
(``AgentConfig.record_sampling``); the trainer distribution is the unmasked current
model at temperature one. That mismatch is an approximation, stated in A5 §4, and it
pulls the trainer's mass toward what the grammar allows.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def klpo_token_loss(
    log_probs: Tensor,
    sampler_log_probs: Tensor,
    rewards: Tensor,
    policy_mask: Tensor,
    *,
    mc_log_probs: Tensor,
    beta: float = 0.1,
) -> tuple[Tensor, dict[str, Tensor]]:
    """The default KLPO update on padded per-episode tensors.

    ``log_probs`` ``[B, T]`` are the current model's log-probabilities of the sampled
    tokens (with autograd); ``sampler_log_probs`` ``[B, T]`` the recorded ones;
    ``rewards`` ``[B]`` terminal; ``policy_mask`` ``[B, T]`` marks tokens the sampler
    drew; ``mc_log_probs`` ``[B, T, M]`` the current model's log-probabilities at the
    recorded auxiliary draws. Sums over tokens, averages over episodes, no length
    normalisation, exactly as the reference contract.
    """
    if beta < 0 or not math.isfinite(beta):
        raise ValueError("beta must be finite and non-negative")
    if log_probs.shape != sampler_log_probs.shape or log_probs.shape != policy_mask.shape:
        raise ValueError("log_probs, sampler_log_probs and policy_mask must share [B, T]")
    if (
        mc_log_probs.dim() != 3
        or mc_log_probs.shape[:2] != log_probs.shape
        or mc_log_probs.shape[2] < 1
    ):
        raise ValueError("mc_log_probs must be [B, T, M] with M >= 1")
    if rewards.shape != (log_probs.shape[0],):
        raise ValueError("rewards must be [B]")
    mask = policy_mask.bool()
    if not bool(mask.any()):
        raise ValueError("no policy tokens to learn from")
    current = log_probs.float()
    sampler = sampler_log_probs.float().masked_fill(~mask, 0.0)
    with torch.no_grad():
        ell = (current - sampler).masked_fill(~mask, 0.0)
        h = (rewards.float()[:, None] - beta * ell).masked_fill(~mask, 0.0)
        local_kl = (sampler - current).masked_fill(~mask, 0.0)  # log q(a) - log p(a), one draw
    correction = mc_log_probs.float().mean(-1)
    z = (current - correction).masked_fill(~mask, 0.0)
    loss = -(h * z).sum(-1).mean()
    if not torch.isfinite(loss):
        raise ValueError("non-finite KLPO loss")
    return loss, {
        "return_coefficient": h,
        "log_ratio": ell,
        "mean_log_ratio": ell.sum() / mask.sum(),
        "sampled_action_kl": local_kl.sum(-1),
        "policy_tokens": mask.sum(-1),
        "mc_samples": torch.tensor(mc_log_probs.shape[-1]),
    }


def episode_tensors(
    episodes: list[dict], *, pad_id: int, device: torch.device | str = "cpu"
) -> dict[str, Tensor]:
    """Pad recorded episodes into the tensors the loss consumes.

    Each episode: ``ids`` (the exact token stream the loop fed), ``sampled`` (records
    ``{position, token, logq, mc_ids, mc_logq}`` for every token the sampler drew) and
    ``reward``. Position ``i`` is predicted by the logits at ``i - 1``.
    """
    if not episodes:
        raise ValueError("no episodes")
    lengths = [len(e["ids"]) for e in episodes]
    width = max(lengths)
    draws = {len(r["mc_ids"]) for e in episodes for r in e["sampled"]}
    if len(draws) > 1:
        raise ValueError("all records must carry the same number of auxiliary draws")
    m = draws.pop() if draws else 1
    b = len(episodes)
    ids = torch.full((b, width), pad_id, dtype=torch.long)
    mask = torch.zeros((b, width), dtype=torch.bool)
    sampler = torch.zeros((b, width), dtype=torch.float32)
    mc_ids = torch.zeros((b, width, m), dtype=torch.long)
    rewards = torch.zeros(b, dtype=torch.float32)
    for row, episode in enumerate(episodes):
        ids[row, : lengths[row]] = torch.tensor(episode["ids"], dtype=torch.long)
        rewards[row] = float(episode["reward"])
        for record in episode["sampled"]:
            position = int(record["position"])
            if not 1 <= position < lengths[row] or episode["ids"][position] != record["token"]:
                raise ValueError("sampling record does not match the token stream")
            mask[row, position] = True
            sampler[row, position] = float(record["logq"])
            mc_ids[row, position] = torch.tensor(record["mc_ids"], dtype=torch.long)
    return {
        "ids": ids.to(device),
        "policy_mask": mask.to(device),
        "sampler_log_probs": sampler.to(device),
        "mc_ids": mc_ids.to(device),
        "rewards": rewards.to(device),
        "lengths": torch.tensor(lengths),
    }


def gather_current(logits: Tensor, ids: Tensor, mc_ids: Tensor) -> tuple[Tensor, Tensor]:
    """Current log-probabilities of the stream's tokens and of the auxiliary draws.

    ``logits`` ``[B, T, V]``; token ``i`` is scored by the logits at ``i - 1``, so the
    returned tensors are aligned to positions ``1 .. T-1`` (position 0 is never a policy
    token).
    """
    log_probs = F.log_softmax(logits.float(), dim=-1)
    shifted = log_probs[:, :-1]  # predicts ids[:, 1:]
    current = torch.zeros(ids.shape, dtype=torch.float32, device=ids.device)
    current[:, 1:] = shifted.gather(-1, ids[:, 1:, None]).squeeze(-1)
    mc = torch.zeros((*ids.shape, mc_ids.shape[-1]), dtype=torch.float32, device=ids.device)
    mc[:, 1:] = shifted.gather(-1, mc_ids[:, 1:])
    return current, mc


def klpo_update(
    model,
    episodes: list[dict],
    *,
    pad_id: int,
    steps: int,
    beta: float = 0.1,
    lr: float = 5e-4,
    batch_size: int = 8,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> dict:
    """``steps`` KLPO updates over recorded episodes; a fresh AdamW each call."""
    if steps < 1:
        return {"steps": 0, "losses": [], "mean_return_coefficient": None}
    usable = [e for e in episodes if e["sampled"]]
    if not usable:
        return {
            "steps": 0,
            "losses": [],
            "mean_return_coefficient": None,
            "reason": "no sampled tokens",
        }
    tensors = episode_tensors(usable, pad_id=pad_id, device=device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0
    )
    generator = torch.Generator().manual_seed(seed)
    model.train()
    losses, coefficients, ratios = [], [], []
    n = tensors["ids"].shape[0]
    for _ in range(steps):
        rows = torch.randperm(n, generator=generator)[: min(batch_size, n)]
        ids = tensors["ids"][rows]
        logits = model(ids, return_mtp=False).logits
        current, mc = gather_current(logits, ids, tensors["mc_ids"][rows])
        loss, stats = klpo_token_loss(
            current,
            tensors["sampler_log_probs"][rows],
            tensors["rewards"][rows],
            tensors["policy_mask"][rows],
            mc_log_probs=mc,
            beta=beta,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        mask = tensors["policy_mask"][rows]
        coefficients.append(float(stats["return_coefficient"].sum() / mask.sum()))
        ratios.append(float(stats["mean_log_ratio"]))
    model.eval()
    return {
        "steps": steps,
        "episodes": len(usable),
        "policy_tokens": int(tensors["policy_mask"].sum()),
        "losses": losses,
        "mean_return_coefficient": sum(coefficients) / len(coefficients),
        "mean_log_ratio_last": ratios[-1],
        "beta": beta,
        "lr": lr,
    }
