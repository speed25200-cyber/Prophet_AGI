"""Experimental activation-weighted shared projection fitting.

No model/config default uses this module. A lower local quadratic error does not
establish language quality after composing the fitted projections.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


@torch.no_grad()
def fit_shared_projection(
    weights: list[Tensor], moments: list[Tensor], *, ridge: float = 0.01
) -> tuple[Tensor, dict[str, float]]:
    """Fit sum ||(W-W_i) X_i||² + lambda ||W-mean(W_i)||².

    ``moments[i]`` is X_i @ X_i.T (features by observations). Inputs and
    linear algebra are FP64; the returned fitted weight is FP32 for the scout.
    Ridge is scaled by the mean diagonal of the summed second moments. It also
    makes unseen directions revert to the mean rather than arbitrary solutions.
    """
    if len(weights) < 2 or len(weights) != len(moments):
        raise ValueError("need at least two matching weight/moment pairs")
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and positive")
    shape = weights[0].shape
    if len(shape) != 2:
        raise ValueError("weights must be matrices")
    device = weights[0].device
    for weight, moment in zip(weights, moments, strict=True):
        if weight.shape != shape or moment.shape != (shape[1], shape[1]):
            raise ValueError("incompatible projection or moment dimensions")
        if weight.device != device or moment.device != device:
            raise ValueError("all inputs must be on the same device")
        if moment.dtype != torch.float64:
            raise ValueError("moments must be accumulated in FP64")
        if not torch.isfinite(weight).all() or not torch.isfinite(moment).all():
            raise ValueError("nonfinite fit inputs")
        if not torch.allclose(moment, moment.T, atol=1e-9, rtol=1e-9):
            raise ValueError("second moments must be symmetric")
    matrices = [w.double() for w in weights]
    mean = torch.stack(matrices).mean(0)
    moment_sum = torch.stack(moments).sum(0)
    scale = float(moment_sum.diagonal().mean())
    if scale <= 0 or not math.isfinite(scale):
        raise ValueError("calibration has no positive activation energy")
    penalty = ridge * scale
    system = moment_sum.clone()
    system.diagonal().add_(penalty)
    rhs = sum(w @ h for w, h in zip(matrices, moments, strict=True)) + penalty * mean
    factor, info = torch.linalg.cholesky_ex(system)
    if int(info) != 0:
        raise ValueError("regularized second moments are not positive definite")
    fitted = torch.cholesky_solve(rhs.T, factor).T

    def error(value):
        return sum(
            float(((value - w) @ h * (value - w)).sum())
            for w, h in zip(matrices, moments, strict=True)
        )

    before = error(mean)
    after = error(fitted)
    regularizer = penalty * float((fitted - mean).square().sum())
    objective = after + regularizer
    if not all(math.isfinite(x) for x in (before, after, objective)):
        raise ValueError("nonfinite fit objective")
    if min(before, after) < -1e-8 * max(1, before):
        raise ValueError("negative second-moment quadratic")
    if objective > before + 1e-8 * max(1, before):
        raise ValueError("fit fails the mean-baseline objective bound")
    rounded = fitted.float()
    rounded_error = error(rounded.double())
    return rounded, {
        "ridge_fraction": ridge,
        "ridge_penalty": penalty,
        "mean_data_error": before,
        "fit_data_error_fp64": after,
        "fit_regularized_objective_fp64": objective,
        "fit_data_error_fp32": rounded_error,
        "relative_weight_change": float((fitted - mean).norm() / mean.norm().clamp_min(1e-30)),
    }
