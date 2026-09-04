#!/usr/bin/env python3
"""Expressivity probe: does the delta-rule write strength decide what is learnable?

An expressivity limit does not show up in a loss curve. A model missing an entire class of
functions trains perfectly normally and simply never solves those problems, and nothing in
the metrics says why. This probe is the cheap way to find one.

The claim it tests: the state transition of a delta-rule layer is
``alpha * (I - beta * k k^T)``. With ``beta`` in (0,1) every eigenvalue stays strictly
positive, so no product of such transitions can change sign -- and parity is exactly a
sign-flip problem. Widening to (0,2) lets the transition reflect, and parity comes back
within reach.

Trained at length 32 and evaluated at 128, because length generalisation is where a model
that learned the *rule* separates from one that fitted the training lengths.

Measured on this implementation:

    | beta range | layers | acc @32 | acc @128 |
    |------------|-------:|--------:|---------:|
    | (0, 1)     |      1 |   0.531 |    0.508 |
    | (0, 1)     |      2 |   0.521 |    0.504 |
    | (0, 2)     |      1 |   1.000 |    0.996 |

Chance against near-perfect, for one multiplication. ``MixerConfig.linear_beta_max``
defaults to 2.0 and :meth:`ProphetConfig.design_warnings` refuses 1.0 silently.

    python scripts/parity_probe.py
"""
import sys; sys.path.insert(0, '/home/user/Prophet_AGI')
import torch, torch.nn.functional as F
from torch import nn
from prophet.modeling.layers import GatedDeltaNet, RMSNorm
torch.set_num_threads(8)


class ParityNet(nn.Module):
    def __init__(self, d=64, layers=2, beta_max=2.0):
        super().__init__()
        self.embed = nn.Embedding(2, d)
        self.blocks = nn.ModuleList(
            GatedDeltaNet(d, n_heads=2, head_dim=32, expand=1.0, conv_kernel=1,
                          beta_max=beta_max)
            for _ in range(layers)
        )
        self.norms = nn.ModuleList(RMSNorm(d) for _ in range(layers))
        self.out_norm = RMSNorm(d)
        self.head = nn.Linear(d, 2)

    def forward(self, bits):
        h = self.embed(bits)
        for norm, block in zip(self.norms, self.blocks):
            h = h + block(norm(h))
        return self.head(self.out_norm(h))


def batch(n, L, gen):
    bits = torch.randint(0, 2, (n, L), generator=gen)
    return bits, torch.cumsum(bits, dim=1) % 2


def run(beta_max, layers, seed, steps=400, train_len=32, test_len=128):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 500)
    model = ParityNet(layers=layers, beta_max=beta_max)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)

    model.train()
    for step in range(steps):
        bits, target = batch(32, train_len, gen)
        loss = F.cross_entropy(model(bits).reshape(-1, 2), target.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    scores = {}
    with torch.no_grad():
        for name, L in (("train_len", train_len), ("4x_len", test_len)):
            bits, target = batch(256, L, torch.Generator().manual_seed(seed + 999))
            pred = model(bits).argmax(-1)
            scores[name] = float((pred == target).float().mean().item())
    return scores


print("Parity, trained at length 32, evaluated at 32 and 128")
print()
print("| beta range | layers | seed | acc @32 | acc @128 |")
print("|---|---:|---:|---:|---:|")
summary = {}
for beta_max, label in ((1.0, "(0,1)"), (2.0, "(0,2)")):
    for layers in (1, 2):
        for seed in (0, 1):
            s = run(beta_max, layers, seed)
            print("| %s | %d | %d | %.3f | %.3f |" % (label, layers, seed, s["train_len"], s["4x_len"]), flush=True)
            summary.setdefault((label, layers), []).append(s["4x_len"])
print()
for key, vals in summary.items():
    print("%s, %d layer(s): mean acc @128 = %.3f" % (key[0], key[1], sum(vals)/len(vals)))
