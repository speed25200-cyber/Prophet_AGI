"""Descriptions of the open models Prophet can be converted from.

The mixed path chosen for this project trains Prophet-mini from random initialisation —
the honest existence proof of the architecture — and produces Prophet-main by converting
an open donor. Conversion inherits the donor's pretraining (tens of trillions of tokens)
and spends our budget only on the architecture change, which is the only way the compute
arithmetic in ``docs/00_PROBLEM_LANDSCAPE.md`` permits a competitive main model.

Licence is the first-order constraint, not an afterthought. A donor whose terms bind
derivatives makes the resulting model un-releasable, exactly as track R10 found for
Gemma-generated data, so :data:`DONORS` records the licence and
:func:`assert_donor_is_usable` refuses the ones that do not permit permissive release.

.. warning::
   The architecture figures below were written while huggingface.co was unreachable from
   the build environment, so they are **unverified**. Every donor carries
   ``verified=False`` until :mod:`scripts.verify_donors` has confirmed it against the
   Hub's ``config.json``, and :func:`load_donor` refuses to convert an unverified spec
   without an explicit override.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["DonorSpec", "DONORS", "assert_donor_is_usable", "get_donor", "LicenceProblem"]


class LicenceProblem(RuntimeError):
    """Raised when a donor's terms would prevent releasing the converted model."""


@dataclass(frozen=True)
class DonorSpec:
    """Architecture and licence of a candidate donor.

    Field names follow the HuggingFace ``config.json`` conventions so a verification
    script can compare them field by field.
    """

    name: str
    hf_id: str
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    ffn_hidden: int
    vocab_size: int
    tie_word_embeddings: bool
    rope_theta: float
    """The donor's rotary base. Copying attention weights under a different base changes
    what every copied head computes at every position beyond a few hundred tokens."""
    license: str
    permits_permissive_release: bool
    """Whether a converted derivative may itself be released under Apache-2.0."""
    naming_constraint: str = ""
    """A non-empty value means the derived model's name is constrained by the licence."""
    verified: bool = False
    """Set only once the figures have been checked against the Hub."""
    notes: str = ""

    @property
    def params_estimate(self) -> int:
        """Rough parameter count, used to sanity-check a spec before downloading GBs."""
        embed = self.vocab_size * self.d_model * (1 if self.tie_word_embeddings else 2)
        per_layer = (
            self.d_model * self.n_heads * self.head_dim          # q
            + 2 * self.d_model * self.n_kv_heads * self.head_dim  # k, v
            + self.n_heads * self.head_dim * self.d_model         # o
            + 3 * self.d_model * self.ffn_hidden                  # SwiGLU
            + 2 * self.d_model                                    # norms
        )
        return embed + self.n_layers * per_layer


#: Candidate donors. Figures unverified — see the module warning.
DONORS: dict[str, DonorSpec] = {
    "qwen3-1.7b": DonorSpec(
        name="Qwen3-1.7B", hf_id="Qwen/Qwen3-1.7B",
        n_layers=28, d_model=2048, n_heads=16, n_kv_heads=8, head_dim=128,
        ffn_hidden=6144, vocab_size=151936, tie_word_embeddings=True, rope_theta=1000000.0,
        license="Apache-2.0", permits_permissive_release=True,
        notes="The reference competitor. head_dim 128 matches Prophet's attention slots.",
    ),
    "qwen3-4b": DonorSpec(
        name="Qwen3-4B", hf_id="Qwen/Qwen3-4B",
        n_layers=36, d_model=2560, n_heads=32, n_kv_heads=8, head_dim=128,
        ffn_hidden=9728, vocab_size=151936, tie_word_embeddings=True, rope_theta=1000000.0,
        license="Apache-2.0", permits_permissive_release=True,
        notes="Stronger donor, but 36 layers of 2560 width is a larger conversion job.",
    ),
    "qwen3-0.6b": DonorSpec(
        name="Qwen3-0.6B", hf_id="Qwen/Qwen3-0.6B",
        n_layers=28, d_model=1024, n_heads=16, n_kv_heads=8, head_dim=128,
        ffn_hidden=3072, vocab_size=151936, tie_word_embeddings=True, rope_theta=1000000.0,
        license="Apache-2.0", permits_permissive_release=True,
        notes="Cheapest conversion target; useful for rehearsing the recipe end to end.",
    ),
    "smollm3-3b": DonorSpec(
        name="SmolLM3-3B", hf_id="HuggingFaceTB/SmolLM3-3B",
        n_layers=36, d_model=2048, n_heads=16, n_kv_heads=4, head_dim=128,
        ffn_hidden=11008, vocab_size=128256, tie_word_embeddings=True, rope_theta=5000000.0,
        license="Apache-2.0", permits_permissive_release=True,
        notes="Fully open training data, which makes contamination auditing tractable.",
    ),
    "llama-3.2-1b": DonorSpec(
        name="Llama-3.2-1B", hf_id="meta-llama/Llama-3.2-1B",
        n_layers=16, d_model=2048, n_heads=32, n_kv_heads=8, head_dim=64,
        ffn_hidden=8192, vocab_size=128256, tie_word_embeddings=True, rope_theta=500000.0,
        license="Llama 3.2 Community License", permits_permissive_release=False,
        naming_constraint="the derived model's name must begin with 'Llama'",
        notes="Usable, but the licence follows the derivative. head_dim 64, not 128.",
    ),
}


def get_donor(key: str) -> DonorSpec:
    if key not in DONORS:
        raise KeyError(f"unknown donor {key!r}; known: {sorted(DONORS)}")
    return DONORS[key]


def assert_donor_is_usable(spec: DonorSpec, *, allow_restricted: bool = False) -> None:
    """Refuse a donor whose licence would compromise the release.

    This is deliberately a hard failure. A licence problem found after conversion cannot
    be fixed by anything short of redoing the conversion from a different donor, and the
    cost of noticing late is the entire budget spent on it.
    """
    if not spec.permits_permissive_release and not allow_restricted:
        raise LicenceProblem(
            f"{spec.name} is under {spec.license}, which follows the derivative: a "
            f"converted Prophet could not be released under Apache-2.0"
            + (f" and {spec.naming_constraint}" if spec.naming_constraint else "")
            + ". Pass allow_restricted=True only if that is an accepted outcome."
        )
