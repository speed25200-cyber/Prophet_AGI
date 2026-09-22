"""Independent least-squares oracles for activation-weighted sharing."""

import pytest
import torch

from prophet.convert.shared_fit import fit_shared_projection


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fit_matches_augmented_design_lstsq_and_beats_weight_mean(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires actual CUDA for the FP64 fit oracle")
    torch.manual_seed(31)
    weights = [torch.randn(2, 3, dtype=torch.float64) for _ in range(2)]
    xs = [torch.randn(3, 13, dtype=torch.float64), torch.randn(3, 17, dtype=torch.float64) * 3]
    moments = [x @ x.T for x in xs]
    actual, report = fit_shared_projection(
        [w.to(device) for w in weights], [h.to(device) for h in moments], ridge=0.03
    )
    mean = torch.stack(weights).mean(0)
    penalty = 0.03 * sum(moments).diagonal().mean()
    design = torch.cat([x.T for x in xs] + [penalty.sqrt() * torch.eye(3, dtype=torch.float64)])
    targets = torch.cat(
        [(w @ x).T for w, x in zip(weights, xs, strict=True)] + [penalty.sqrt() * mean.T]
    )
    expected = torch.linalg.lstsq(design, targets).solution.T
    torch.testing.assert_close(actual.cpu().double(), expected, atol=1e-6, rtol=1e-6)
    assert report["fit_regularized_objective_fp64"] < report["mean_data_error"]


def test_unobserved_direction_retains_mean_and_chunked_moments_resume_exactly():
    weights = [torch.tensor([[1.0, 3.0]]), torch.tensor([[5.0, 9.0]])]
    # Integer inputs make the independently grouped FP64 accumulation exact.
    chunks = [
        torch.tensor([[1.0, 2.0], [0.0, 0.0]], dtype=torch.float64),
        torch.tensor([[3.0, 4.0], [0.0, 0.0]], dtype=torch.float64),
    ]
    full = torch.cat(chunks, dim=1)
    whole = full @ full.T
    saved = (chunks[0] @ chunks[0].T).clone()
    resumed = saved + chunks[1] @ chunks[1].T
    torch.testing.assert_close(resumed, whole, atol=0, rtol=0)
    result, _ = fit_shared_projection(weights, [whole, 2 * resumed])
    assert result[0, 1] == 6  # unseen second coordinate stays at the mean
    assert result[0, 0] > 3  # higher-energy second layer gets greater influence


def test_refuses_invalid_covariance_and_nonfinite_inputs():
    weights = [torch.ones(2, 2)] * 2
    moment = torch.eye(2, dtype=torch.float64)
    for bad in (moment.float(), moment * float("nan"), torch.zeros_like(moment), -moment):
        with pytest.raises(ValueError):
            fit_shared_projection(weights, [bad, bad])
    with pytest.raises(ValueError):
        fit_shared_projection(weights, [moment, moment], ridge=0)


def test_atomic_moment_snapshot_restricted_reload_preserves_resume(tmp_path):
    from scripts.probe_qwen_covariance import save_moments

    contract = {"source": "frozen", "ridge": 0.01}
    first = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    second = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float64)
    moments = {"layer:projection": first.T @ first}
    checkpoint = tmp_path / "moments.pt"
    save_moments(checkpoint, contract, 1, moments)
    restored = torch.load(checkpoint, weights_only=True)
    assert restored["contract"] == contract and restored["next_document"] == 1
    restored["moments"]["layer:projection"].addmm_(second.T, second)
    continuous = moments["layer:projection"] + second.T @ second
    torch.testing.assert_close(restored["moments"]["layer:projection"], continuous, atol=0, rtol=0)
    save_moments(checkpoint, contract, 2, restored["moments"])
    assert torch.load(checkpoint, weights_only=True)["next_document"] == 2
    assert not checkpoint.with_suffix(".tmp").exists()
