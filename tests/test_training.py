"""Tests for optimisation, scheduling, checkpointing and the training loop.

The headline test is :func:`test_resumed_run_matches_uninterrupted_run`. A Colab A100
session ends without warning, so over a multi-week run the loop will be interrupted
dozens of times; if resume is even slightly wrong, the result is not a crash but a
quietly worse model.
"""

from __future__ import annotations

import math

import pytest
import torch

from prophet.config import (
    FeedForwardConfig,
    FrontendConfig,
    HeadsConfig,
    MixerConfig,
    ProphetConfig,
    RecurrentCoreConfig,
)
from prophet.data.streaming import StreamingLoader, sources_from_iterables
from prophet.modeling.model import ProphetModel
from prophet.train.checkpoint import CheckpointError, CheckpointManager
from prophet.train.loop import TrainConfig, Trainer
from prophet.train.loss import compute_loss
from prophet.train.optim import (
    Muon,
    build_optimizers,
    build_param_groups,
    newton_schulz_orthogonalise,
)
from prophet.train.schedule import CosineSchedule, WSDSchedule

VOCAB = 64


def tiny_model_config() -> ProphetConfig:
    return ProphetConfig(
        name="train-test",
        d_model=64,
        max_seq_len=64,
        frontend=FrontendConfig(vocab_size=VOCAB),
        mixer=MixerConfig(
            pattern=["swa", "full_attn"], n_heads=4, n_kv_heads=2, head_dim=16,
            sliding_window=16, linear_heads=2, linear_head_dim=16,
        ),
        recurrent=RecurrentCoreConfig(
            enabled=True, prelude_layers=1, core_layers=1, coda_layers=1,
            core_pattern=["gdn"], train_loop_min=1, train_loop_max=2,
            truncated_backprop_steps=2,
        ),
        ffn=FeedForwardConfig(kind="dense", hidden_mult=2.0),
        heads=HeadsConfig(),
    )


def make_loader(seed: int = 0) -> StreamingLoader:
    docs = {"a": (1.0, [[(i * 7 + j) % VOCAB for j in range(32)] for i in range(200)])}
    return StreamingLoader(sources_from_iterables(docs), seq_len=32, batch_size=2, seed=seed)


# --------------------------------------------------------------------------------------
# Muon
# --------------------------------------------------------------------------------------


def test_newton_schulz_pushes_singular_values_toward_one():
    """The whole point: spread the update across the spectrum instead of a few directions.

    Five steps of the tuned quintic do not fully orthogonalise an ill-conditioned matrix
    -- they are not meant to. What they do is collapse the condition number by more than
    an order of magnitude, which is what makes the update move in directions the raw
    gradient barely touches.
    """
    torch.manual_seed(0)
    g = torch.randn(64, 32)
    g = g @ torch.diag(torch.logspace(0, -3, 32))  # condition number ~1000
    before = torch.linalg.svdvals(g)
    after = torch.linalg.svdvals(newton_schulz_orthogonalise(g).float())

    assert before.max() / before.min() > 100
    assert after.max() / after.min() < before.max() / before.min() / 20
    assert after.median() > 0.6
    assert after.max() < 1.5


def test_newton_schulz_handles_both_orientations():
    torch.manual_seed(0)
    for shape in ((64, 16), (16, 64)):
        out = newton_schulz_orthogonalise(torch.randn(*shape))
        assert out.shape == shape
        assert torch.isfinite(out).all()


def test_newton_schulz_rejects_non_matrices():
    with pytest.raises(ValueError, match="2-D"):
        newton_schulz_orthogonalise(torch.randn(8))


def test_muon_reduces_a_quadratic_objective():
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(32, 16))
    target = torch.randn(32, 16)
    opt = Muon([w], lr=0.05, weight_decay=0.0)
    first = last = None
    for _ in range(80):
        loss = (w - target).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
        last = loss.item()
    assert last < first * 0.5


def test_muon_rejects_non_matrix_parameters():
    """Routing a norm or bias to Muon is a silent mistake if it is not caught here."""
    p = torch.nn.Parameter(torch.randn(8))
    p.grad = torch.randn(8)
    with pytest.raises(ValueError, match="route 1-D"):
        Muon([p], lr=0.01).step()


def test_router_and_embeddings_never_go_to_muon():
    cfg = tiny_model_config()
    cfg.ffn = FeedForwardConfig(kind="moe", n_experts=4, n_experts_per_token=2,
                                expert_hidden_mult=0.5, moe_first_dense_layers=0)
    model = ProphetModel(cfg)
    groups = build_param_groups(model)
    muon_ids = {id(p) for p in groups["muon"]}
    for name, p in model.named_parameters():
        if any(k in name.lower() for k in ("embed", "router", "norm", "lm_head")):
            assert id(p) not in muon_ids, f"{name} was routed to Muon"


def test_every_parameter_is_assigned_exactly_once():
    model = ProphetModel(tiny_model_config())
    groups = build_param_groups(model)
    assigned = sum(len(v) for v in groups.values())
    unique = {id(p) for p in model.parameters() if p.requires_grad}
    assert assigned == len(unique)


def test_mup_scales_learning_rates_with_width():
    cfg = tiny_model_config()
    cfg.mup_base_width = 32
    model = ProphetModel(cfg)
    _, _ = build_optimizers(model)
    scaled, _ = build_optimizers(model, muon_lr=0.02, mup_base_width=32, d_model=64)
    assert scaled[0].param_groups[0]["lr"] == pytest.approx(0.01)


# --------------------------------------------------------------------------------------
# Schedules
# --------------------------------------------------------------------------------------


def test_wsd_has_three_phases_in_order():
    s = WSDSchedule(peak_lr=1e-3, total_steps=10_000, warmup_frac=0.02, decay_frac=0.18)
    assert s.phase_at(0) == "warmup"
    assert s.phase_at(s.warmup_steps) == "plateau"
    assert s.phase_at(s.decay_start) == "decay"
    assert s.warmup_steps + s.plateau_steps + s.decay_steps == s.total_steps


def test_wsd_plateau_is_exactly_constant():
    """The property that makes plateau checkpoints reusable across anneal branches."""
    s = WSDSchedule(peak_lr=3e-3, total_steps=10_000)
    mid = range(s.warmup_steps, s.decay_start, 137)
    assert len({s.lr_at(i) for i in mid}) == 1


def test_wsd_warmup_is_monotone_and_never_zero():
    s = WSDSchedule(peak_lr=1e-3, total_steps=1000)
    lrs = [s.lr_at(i) for i in range(s.warmup_steps)]
    assert lrs[0] > 0
    assert all(b >= a for a, b in zip(lrs, lrs[1:]))


def test_wsd_decays_to_zero_by_the_end():
    s = WSDSchedule(peak_lr=1e-3, total_steps=1000, final_lr_frac=0.0)
    assert s.lr_at(s.total_steps - 1) < 1e-3 * 0.05


def test_wsd_rejects_a_schedule_with_no_plateau():
    with pytest.raises(ValueError, match="no plateau"):
        WSDSchedule(peak_lr=1e-3, total_steps=1000, warmup_frac=0.5, decay_frac=0.6)


@pytest.mark.parametrize("shape", ["linear", "cosine", "one_minus_sqrt"])
def test_all_decay_shapes_are_monotone(shape):
    s = WSDSchedule(peak_lr=1e-3, total_steps=2000, decay_shape=shape)
    lrs = [s.lr_at(i) for i in range(s.decay_start, s.total_steps)]
    assert all(b <= a + 1e-12 for a, b in zip(lrs, lrs[1:]))


def test_one_minus_sqrt_stays_higher_than_linear_early_in_decay():
    """Why it is the default: more time at a useful learning rate before the drop."""
    common = dict(peak_lr=1e-3, total_steps=2000)
    sq = WSDSchedule(**common, decay_shape="one_minus_sqrt")
    lin = WSDSchedule(**common, decay_shape="linear")
    quarter = sq.decay_start + sq.decay_steps // 4
    assert sq.lr_at(quarter) < lin.lr_at(quarter)


def test_cosine_baseline_still_works():
    s = CosineSchedule(peak_lr=1e-3, total_steps=1000)
    assert s.lr_at(0) > 0
    assert s.lr_at(999) < s.lr_at(500) < s.lr_at(s.warmup_steps)


# --------------------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------------------


def test_checkpoint_roundtrip(tmp_path):
    cm = CheckpointManager(tmp_path)
    cm.save({"w": torch.arange(10)}, step=5)
    state, meta = cm.load_latest()
    assert meta.step == 5
    assert torch.equal(state["w"], torch.arange(10))


def test_slots_alternate_so_a_torn_write_cannot_destroy_both(tmp_path):
    cm = CheckpointManager(tmp_path)
    slots = [cm.save({"step": i}, step=i).slot for i in range(1, 5)]
    assert slots == [0, 1, 0, 1]


def test_corrupted_checkpoint_falls_back_to_the_previous_slot(tmp_path):
    """The exact failure this design exists to survive: preemption mid-write."""
    cm = CheckpointManager(tmp_path)
    cm.save({"v": 1}, step=1)
    meta = cm.save({"v": 2}, step=2)
    with cm.slot_path(meta.slot).open("ab") as fh:
        fh.write(b"truncated-write-garbage")

    state, recovered = cm.load_latest()
    assert recovered.step == 1 and state["v"] == 1


def test_all_slots_corrupted_raises_rather_than_returning_junk(tmp_path):
    cm = CheckpointManager(tmp_path)
    cm.save({"v": 1}, step=1)
    cm.save({"v": 2}, step=2)
    for slot in (0, 1):
        with cm.slot_path(slot).open("ab") as fh:
            fh.write(b"garbage")
    with pytest.raises(CheckpointError, match="no intact checkpoint"):
        cm.load_latest()


def test_milestones_are_kept_outside_the_rotation(tmp_path):
    """The WSD branch point must survive rotation: several anneals are launched from it."""
    cm = CheckpointManager(tmp_path, keep_milestones=(2,))
    for i in range(1, 6):
        cm.save({"step": i}, step=i)
    assert cm.milestone_path(2).exists()


def test_verify_reports_slot_health(tmp_path):
    cm = CheckpointManager(tmp_path)
    cm.save({"v": 1}, step=1)
    meta = cm.save({"v": 2}, step=2)
    assert all(cm.verify().values())
    with cm.slot_path(meta.slot).open("ab") as fh:
        fh.write(b"x")
    assert cm.verify()[meta.slot] is False


def test_has_checkpoint_is_false_on_a_fresh_directory(tmp_path):
    assert not CheckpointManager(tmp_path).has_checkpoint()


# --------------------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------------------


def test_loss_terms_are_all_reported():
    torch.manual_seed(0)
    cfg = tiny_model_config()
    cfg.heads = HeadsConfig(n_multi_token_predict=1)
    model = ProphetModel(cfg)
    batch = torch.randint(0, VOCAB, (2, 16))
    terms = compute_loss(model(batch), batch, mtp_weight=0.3, z_loss_weight=1e-4)
    assert terms.mtp is not None and terms.z is not None
    assert {"loss/lm", "loss/mtp", "loss/z", "loss/total", "ppl"} <= terms.metrics.keys()
    assert terms.total.item() >= terms.lm.item()


def test_untrained_loss_is_near_uniform():
    """A sanity check that catches an off-by-one in the target shift, which would
    otherwise show up as a suspiciously good initial loss."""
    torch.manual_seed(0)
    model = ProphetModel(tiny_model_config())
    batch = torch.randint(0, VOCAB, (4, 32))
    lm = compute_loss(model(batch), batch, z_loss_weight=0.0).lm.item()
    assert abs(lm - math.log(VOCAB)) < 0.6, f"expected ~{math.log(VOCAB):.2f}, got {lm:.2f}"


# --------------------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------------------


def test_training_reduces_loss_on_a_learnable_task(tmp_path):
    torch.manual_seed(0)
    trainer = Trainer(
        ProphetModel(tiny_model_config()),
        make_loader(),
        TrainConfig(total_steps=60, log_every=1000, checkpoint_every=0,
                    peak_lr_muon=0.01, checkpoint_dir=str(tmp_path)),
        model_config=tiny_model_config(),
        on_log=lambda m: None,
    )
    history = trainer.train()
    assert history[-1].loss < history[0].loss * 0.5


def test_resumed_run_matches_uninterrupted_run(tmp_path):
    """A resumed run must be indistinguishable from one that was never interrupted.

    This covers model weights, optimiser state, the schedule position, the data cursor,
    and the RNG that samples recurrence depth. Getting any one of them wrong degrades
    the model without producing an error.
    """
    def fresh(directory):
        torch.manual_seed(1234)
        model = ProphetModel(tiny_model_config())
        cfg = TrainConfig(total_steps=20, log_every=1000, checkpoint_every=10,
                          peak_lr_muon=0.01, checkpoint_dir=str(directory))
        return Trainer(model, make_loader(), cfg, model_config=tiny_model_config(),
                       on_log=lambda m: None)

    reference = fresh(tmp_path / "ref")
    reference.train()
    expected = {k: v.clone() for k, v in reference.model.state_dict().items()}

    interrupted = fresh(tmp_path / "run")
    interrupted.train(max_steps=10)

    resumed = fresh(tmp_path / "run")
    assert resumed.maybe_resume()
    assert resumed.step == 10
    resumed.train()

    for key, value in resumed.model.state_dict().items():
        assert torch.allclose(value, expected[key], atol=1e-6), f"diverged at {key}"


def test_resume_restores_the_token_counter(tmp_path):
    trainer = Trainer(
        ProphetModel(tiny_model_config()), make_loader(),
        TrainConfig(total_steps=20, log_every=1000, checkpoint_every=5,
                    checkpoint_dir=str(tmp_path)),
        model_config=tiny_model_config(), on_log=lambda m: None,
    )
    trainer.train(max_steps=10)
    tokens = trainer.tokens_seen

    resumed = Trainer(
        ProphetModel(tiny_model_config()), make_loader(),
        TrainConfig(total_steps=20, log_every=1000, checkpoint_every=5,
                    checkpoint_dir=str(tmp_path)),
        model_config=tiny_model_config(), on_log=lambda m: None,
    )
    resumed.maybe_resume()
    assert resumed.tokens_seen == tokens


def test_no_resume_on_an_empty_directory(tmp_path):
    trainer = Trainer(
        ProphetModel(tiny_model_config()), make_loader(),
        TrainConfig(total_steps=5, checkpoint_dir=str(tmp_path)),
        model_config=tiny_model_config(), on_log=lambda m: None,
    )
    assert not trainer.maybe_resume()
    assert trainer.step == 0


def test_gradient_accumulation_matches_the_token_count(tmp_path):
    trainer = Trainer(
        ProphetModel(tiny_model_config()), make_loader(),
        TrainConfig(total_steps=4, grad_accum_steps=3, log_every=1000,
                    checkpoint_every=0, checkpoint_dir=str(tmp_path)),
        model_config=tiny_model_config(), on_log=lambda m: None,
    )
    trainer.train()
    assert trainer.tokens_seen == 4 * 3 * 2 * 32


# --------------------------------------------------------------------------------------
# Ponder loss
# --------------------------------------------------------------------------------------


def _halting_config() -> ProphetConfig:
    cfg = tiny_model_config()
    cfg.recurrent = RecurrentCoreConfig(
        enabled=True, prelude_layers=1, core_layers=1, coda_layers=1,
        core_pattern=["gdn"], default_loop_k=5, halting="ponder",
        halting_loss_weight=0.05, halting_target_steps=3.0,
    )
    return cfg


def test_ponder_loss_reaches_the_halting_head():
    """Without its own objective the halting head gets no gradient at all -- the halting
    distribution does not enter the logits. An untrained head makes depth *look*
    input-dependent while being noise."""
    torch.manual_seed(0)
    model = ProphetModel(_halting_config())
    batch = torch.randint(0, VOCAB, (2, 8))

    terms = compute_loss(
        model(batch), batch, ponder_weight=0.05, ponder_target_steps=3.0,
        project=model._project,
    )
    terms.total.backward()

    grad = model.halt_head[1].weight.grad
    assert grad is not None and grad.abs().sum() > 0


def test_ponder_metrics_are_reported():
    torch.manual_seed(0)
    model = ProphetModel(_halting_config())
    batch = torch.randint(0, VOCAB, (2, 8))
    terms = compute_loss(
        model(batch), batch, ponder_weight=0.05, project=model._project
    )
    assert {"loss/ponder", "loss/ponder_kl", "ponder/expected_depth"} <= terms.metrics.keys()


def test_ponder_is_skipped_when_unweighted():
    torch.manual_seed(0)
    model = ProphetModel(_halting_config())
    batch = torch.randint(0, VOCAB, (2, 8))
    terms = compute_loss(model(batch), batch, ponder_weight=0.0)
    assert terms.ponder is None and "loss/ponder" not in terms.metrics


def test_the_prior_pulls_expected_depth_toward_its_target():
    """Without the prior the head learns to always think as long as it is allowed: more
    computation never hurts the language-modelling loss."""
    from prophet.train.loss import _geometric_prior

    for target in (2.0, 8.0):
        prior = _geometric_prior(16, target, torch.device("cpu"), torch.float32)
        mean = (prior * torch.arange(1, 17, dtype=torch.float32)).sum().item()
        assert prior.sum().item() == pytest.approx(1.0, abs=1e-5)
        assert mean < target + 2.0
    shallow = _geometric_prior(16, 2.0, torch.device("cpu"), torch.float32)
    deep = _geometric_prior(16, 8.0, torch.device("cpu"), torch.float32)
    assert shallow[0] > deep[0]


def test_trainer_picks_up_the_halting_weight_from_the_model_config(tmp_path):
    """The silent failure this prevents: halting enabled architecturally, never trained,
    producing a depth distribution that is pure noise."""
    cfg = _halting_config()
    trainer = Trainer(
        ProphetModel(cfg), make_loader(),
        TrainConfig(total_steps=2, log_every=1000, checkpoint_every=0,
                    checkpoint_dir=str(tmp_path)),
        model_config=cfg, on_log=lambda m: None,
    )
    assert trainer.cfg.ponder_weight == pytest.approx(0.05)
    assert trainer.cfg.ponder_target_steps == pytest.approx(3.0)


def test_training_with_halting_runs_and_reports_depth(tmp_path):
    cfg = _halting_config()
    logged: list = []
    trainer = Trainer(
        ProphetModel(cfg), make_loader(),
        TrainConfig(total_steps=6, log_every=1, checkpoint_every=0,
                    peak_lr_muon=0.01, checkpoint_dir=str(tmp_path)),
        model_config=cfg, on_log=logged.append,
    )
    trainer.train()
    assert logged and any("ponder/expected_depth" in m.extra for m in logged)


def test_wall_clock_deadline_stops_the_run_cleanly(tmp_path):
    """A run past its wall-clock budget stops at a step boundary and can resume."""
    from prophet.train.loop import TrainConfig, Trainer

    cfg = ProphetConfig.from_json("configs/prophet_tiny_smoke.json")
    model = ProphetModel(cfg)
    rows = [[int(x) for x in torch.randint(0, 2048, (32,))] for _ in range(8)]
    loader = StreamingLoader(sources_from_iterables({"a": (1.0, rows)}), seq_len=32, batch_size=1)
    trainer = Trainer(
        model, loader,
        TrainConfig(total_steps=50, seq_len=32, checkpoint_dir=str(tmp_path), device="cpu",
                    max_wall_seconds=0.0),
        model_config=cfg,
    )
    trainer.train()
    assert trainer.step <= 1 and trainer.stop_requested
