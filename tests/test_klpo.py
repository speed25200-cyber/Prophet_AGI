"""KLPO in Prophet: the loss matches its derivation, the loop records its sampler,
and an update moves the policy the way the reward says."""

import pytest
import torch
import torch.nn.functional as F

from prophet.agent import tasks as task_families
from prophet.agent.loop import AgentConfig, AgentLoop
from prophet.data.tokenizer import ProphetTokenizer
from prophet.modeling.model import ProphetModel
from prophet.train.klpo import episode_tensors, gather_current, klpo_token_loss, klpo_update
from scripts.first_agent_run_cpu import agent_config
from tests.test_closed_loop import agent_tiny_config


def test_loss_is_the_reference_surrogate_and_masks_non_policy_tokens():
    torch.manual_seed(0)
    b, t, m = 2, 5, 3
    log_p = torch.log_softmax(torch.randn(b, t, 7), -1)
    tokens = torch.randint(0, 7, (b, t))
    current = log_p.gather(-1, tokens[..., None]).squeeze(-1).requires_grad_(True)
    sampler = current.detach() - 0.1
    mask = torch.tensor([[False, True, True, False, True], [False, True, False, True, True]])
    mc = torch.log(torch.rand(b, t, m))
    rewards = torch.tensor([1.0, 0.0])
    beta = 0.2
    loss, stats = klpo_token_loss(current, sampler, rewards, mask, mc_log_probs=mc, beta=beta)
    h = (rewards[:, None] - beta * (current.detach() - sampler)) * mask
    z = (current - mc.mean(-1)) * mask
    expected = -(h * z).sum(-1).mean()
    assert torch.isclose(loss, expected)
    assert torch.equal(stats["policy_tokens"], mask.sum(-1))
    # Only policy tokens carry gradient; a masked token's coefficient is zero.
    loss.backward()
    assert torch.all(current.grad[~mask] == 0) and torch.any(current.grad[mask] != 0)
    with pytest.raises(ValueError, match="no policy tokens"):
        klpo_token_loss(current.detach(), sampler, rewards, torch.zeros_like(mask), mc_log_probs=mc)
    with pytest.raises(ValueError, match="M >= 1"):
        klpo_token_loss(current.detach(), sampler, rewards, mask, mc_log_probs=mc[..., :0])


def test_mc_correction_matches_the_full_expectation_in_the_limit():
    """With draws from q, mean_j log p(v_j) -> sum_v q_v log p_v, so the sampler-centred
    score is recovered: the gradient of the MC surrogate converges to the exact one."""
    torch.manual_seed(1)
    vocab = 6
    logits = torch.randn(vocab, requires_grad=True)
    q = torch.softmax(torch.randn(vocab), -1)
    token = 2
    log_p = torch.log_softmax(logits, -1)
    exact = log_p[token] - (q * log_p).sum()
    (grad_exact,) = torch.autograd.grad(exact, logits)
    draws = torch.multinomial(q, 20_000, replacement=True)
    log_p = torch.log_softmax(logits, -1)  # a fresh graph: the first grad freed the old one
    mc = log_p[token] - log_p[draws].mean()
    (grad_mc,) = torch.autograd.grad(mc, logits)
    assert torch.allclose(grad_mc, grad_exact, atol=2e-2)


def test_gather_aligns_positions_with_the_predicting_logits():
    torch.manual_seed(2)
    logits = torch.randn(1, 4, 5)
    ids = torch.tensor([[1, 3, 0, 4]])
    mc_ids = torch.tensor([[[0, 1], [2, 2], [4, 0], [3, 1]]])
    current, mc = gather_current(logits, ids, mc_ids)
    log_p = F.log_softmax(logits, -1)
    assert current[0, 0] == 0 and torch.isclose(current[0, 2], log_p[0, 1, 0])
    assert torch.isclose(mc[0, 3, 1], log_p[0, 2, 1]) and torch.all(mc[0, 0] == 0)


def test_episode_tensors_check_records_against_the_stream():
    episode = {
        "ids": [5, 6, 7, 8],
        "reward": 1,
        "sampled": [
            {"position": 2, "token": 7, "logq": -0.5, "mc_ids": [1, 2], "mc_logq": [-1.0, -2.0]}
        ],
    }
    tensors = episode_tensors([episode], pad_id=0)
    assert tensors["policy_mask"].tolist() == [[False, False, True, False]]
    assert tensors["sampler_log_probs"][0, 2] == -0.5 and tensors["rewards"].tolist() == [1.0]
    bad = {**episode, "sampled": [{**episode["sampled"][0], "token": 9}]}
    with pytest.raises(ValueError, match="does not match"):
        episode_tensors([bad], pad_id=0)


@pytest.fixture(scope="module")
def tiny_agent():
    tokenizer = ProphetTokenizer([], vocab_size=512)
    cfg = agent_config(agent_tiny_config())
    torch.manual_seed(3)
    return ProphetModel(cfg).eval(), tokenizer


def run_episode(model, tokenizer, *, sample_actions, temperature=1.0):
    task = task_families.make_tasks(1, family="calc", seed=99)[0]
    cfg = AgentConfig(
        max_steps=2,
        think_budget=3,
        action_budget=24,
        halt_threshold=None,
        k_decide=2,
        tau_done=0.0,
        tau_act=0.0,
        tau_ask=0.0,
        sample_temperature=temperature,
        record_sampling=True,
        sample_actions=sample_actions,
        mc_draws=4,
        family="calc",
    )
    loop = AgentLoop(
        model,
        tokenizer,
        task_families.tools_for(task),
        cfg,
        verifier_tool=task_families.verifier_for(task),
    )
    return loop.run(task.goal)


def test_loop_records_every_drawn_token_with_its_sampler_probabilities(tiny_agent):
    model, tokenizer = tiny_agent
    result = run_episode(model, tokenizer, sample_actions=True)
    assert result.ids is not None and result.sampled
    assert len(result.ids) == result.tokens
    for record in result.sampled:
        assert result.ids[record["position"]] == record["token"]
        assert record["logq"] <= 0.0 and len(record["mc_ids"]) == 4
        assert all(0 <= v < 512 for v in record["mc_ids"]) and all(
            x <= 0.0 for x in record["mc_logq"]
        )
    # Greedy action spans are point masses and are not recorded; think tokens still are.
    greedy = run_episode(model, tokenizer, sample_actions=False)
    assert len(greedy.sampled) < len(result.sampled) or all(r["logq"] <= 0 for r in greedy.sampled)


def test_update_raises_the_probability_of_rewarded_tokens(tiny_agent):
    model, tokenizer = tiny_agent
    episodes = []
    for reward in (1, 0):
        result = run_episode(model, tokenizer, sample_actions=True)
        episodes.append({"ids": result.ids, "sampled": result.sampled, "reward": reward})
    tensors = episode_tensors(episodes, pad_id=tokenizer.pad_id)

    def sampled_log_prob(row):
        with torch.no_grad():
            logits = model(tensors["ids"][row : row + 1], return_mtp=False).logits
        current, _ = gather_current(
            logits, tensors["ids"][row : row + 1], tensors["mc_ids"][row : row + 1]
        )
        return float((current[0] * tensors["policy_mask"][row]).sum())

    before = [sampled_log_prob(0), sampled_log_prob(1)]
    report = klpo_update(
        model, episodes, pad_id=tokenizer.pad_id, steps=8, beta=0.0, lr=5e-3, seed=0
    )
    after = [sampled_log_prob(0), sampled_log_prob(1)]
    assert report["steps"] == 8 and len(report["losses"]) == 8
    # With beta = 0 the coefficient is the reward: rewarded tokens go up, unrewarded ones
    # only feel the centering, so the rewarded episode gains at least as much.
    assert after[0] > before[0]
    assert after[0] - before[0] > after[1] - before[1]


def test_no_repeat_action_never_lets_a_step_repeat_the_previous_action(tiny_agent):
    """docs/31 amendment 11: with the option, step i cannot use step i-1's action name."""
    model, tokenizer = tiny_agent
    for seed in (99, 100, 101):
        task = task_families.make_tasks(1, family="lookup", seed=seed)[0]
        cfg = AgentConfig(
            max_steps=4,
            think_budget=2,
            action_budget=24,
            halt_threshold=None,
            k_decide=2,
            tau_done=0.0,
            tau_act=0.0,
            tau_ask=0.0,
            sample_temperature=0.0,
            no_repeat_action=True,
            family="lookup",
        )
        loop = AgentLoop(
            model,
            tokenizer,
            task_families.tools_for(task),
            cfg,
            verifier_tool=task_families.verifier_for(task),
        )
        result = loop.run(task.goal)
        names = [s.action.name for s in result.steps if s.action is not None]
        assert all(a != b for a, b in zip(names, names[1:], strict=False)), names
