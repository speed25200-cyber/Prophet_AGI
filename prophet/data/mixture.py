"""Declarative data-mixture model and its safety checks.

The mixture is deliberately represented with small dataclasses.  This keeps the
training recipe inspectable, serialisable, and cheap to validate before any corpus is
downloaded.
"""

from __future__ import annotations

import math
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "BLOCKED_LICENSES",
    "MAX_EPOCHS",
    "Mixture",
    "MixtureError",
    "Phase",
    "RELEASE_LICENSES",
    "Source",
    "canonical_license_key",
    "validate_release_license",
]


MAX_EPOCHS = 4.0
"""Maximum planned repetitions of a source before validation fails."""

# Substrings are intentionally lower-case: scripts/verify_datasets.py uses this list
# directly against the licence reported by the Hub.
BLOCKED_LICENSES = (
    "gemma",
    "cc-by-nc",
    "cc by-nc",
    "cc by nc",
    "non-commercial",
    "noncommercial",
    "research-only",
    "research only",
    "all rights reserved",
    "proprietary",
)

# Release validation is deliberately an allowlist, not a denylist. Dataset-card licence
# strings are uncontrolled input and there are too many restrictive or bespoke terms to
# enumerate safely. Canonical keys are produced by ``_license_key``; any new key requires
# a deliberate project review before it is added here.
RELEASE_LICENSES = frozenset(
    {
        "apache20",
        "bsd2clause",
        "bsd3clause",
        "cc010",
        "ccby40",
        "isc",
        "mit",
        "nvidiaopendatapermissive",
        "odcby10",
        "unlicense",
    }
)

_BLOCKED_LICENSE_KEYS = (
    "allrightsreserved",
    "ccbync",
    "gemma",
    "noncommercial",
    "proprietary",
    "researchonly",
)
_REVIEW_LICENSE_KEYS = ("bespoke", "custom", "mixed", "persubset", "review")
_LICENSE_ALIASES = {
    "apachelicense20": "apache20",
    "cc0": "cc010",
    "mitlicense": "mit",
    "odcby": "odcby10",
    "opendatacommonsattributionlicense10": "odcby10",
}


def _license_key(value: str) -> str:
    """Return a separator- and Unicode-insensitive licence-policy key."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def canonical_license_key(value: str) -> str:
    """Canonicalize common equivalent spellings for comparison and policy checks."""

    key = _license_key(value)
    return _LICENSE_ALIASES.get(key, key)


def validate_release_license(
    licence: str,
    *,
    label: str = "source",
    allow_pending_review: bool = False,
) -> None:
    """Apply the project's fail-closed release policy to one licence string."""

    raw = (licence or "").strip()
    key = _license_key(raw)
    if not raw or key in {"unknown", "none", "na", "tbd"}:
        raise MixtureError(f"{label}: licence not established")
    if "gemma" in key:
        raise MixtureError(
            f"{label}: Gemma terms would make the trained model a Model Derivative"
        )
    if any(marker in key for marker in _BLOCKED_LICENSE_KEYS):
        raise MixtureError(f"{label}: restrictive licence is not releasable")
    if any(marker in key for marker in _REVIEW_LICENSE_KEYS):
        if allow_pending_review:
            return
        raise MixtureError(
            f"{label}: licence {raw!r} requires REVIEW before this source can train"
        )
    if canonical_license_key(raw) not in RELEASE_LICENSES:
        raise MixtureError(
            f"{label}: licence {raw!r} is not on the release allowlist; "
            "mark it REVIEW until audited"
        )


def _source_identity(source: Source) -> tuple[str, str, str]:
    """Identify one physical/filtered corpus independently of phase-local aliases."""

    filters = yaml.safe_dump(
        source.filters,
        sort_keys=True,
        allow_unicode=True,
        default_flow_style=True,
    )
    return source.hf_id.strip(), (source.config or "").strip(), filters


class MixtureError(ValueError):
    """A data recipe is unsafe or internally inconsistent."""


@dataclass(slots=True)
class Source:
    """One corpus in a phase.

    ``available_tokens`` is optional because an unknown size is preferable to a made-up
    estimate.  Unknown sizes are surfaced by :meth:`Mixture.unverified_sources`.
    """

    name: str
    hf_id: str
    domain: str
    weight: float
    available_tokens: float | None = None
    license: str = "unknown"
    config: str | None = None
    filters: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def allocated_tokens(self, phase_tokens: float) -> float:
        return phase_tokens * self.weight

    def epochs(self, phase_tokens: float) -> float | None:
        if self.available_tokens is None:
            return None
        return self.allocated_tokens(phase_tokens) / self.available_tokens

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Source:
        values = dict(data)
        # Be liberal when reading hand-written recipes while always writing ``license``.
        if "licence" in values and "license" not in values:
            values["license"] = values.pop("licence")
        values["filters"] = dict(values.get("filters") or {})
        return cls(**values)


@dataclass(slots=True)
class Phase:
    """A contiguous training phase with its own local source distribution."""

    name: str
    weight: float
    sources: list[Source]
    lr_schedule: str = "constant"
    context_len: int = 4096
    purpose: str = ""

    def allocated_tokens(self, total_tokens: float) -> float:
        return total_tokens * self.weight

    def domain_shares(self) -> dict[str, float]:
        shares: defaultdict[str, float] = defaultdict(float)
        for source in self.sources:
            shares[source.domain] += source.weight
        return dict(shares)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Phase:
        values = dict(data)
        values["sources"] = [Source.from_dict(source) for source in values.get("sources", [])]
        return cls(**values)


@dataclass(slots=True)
class Mixture:
    """A complete, budget-scalable pre-training data recipe."""

    name: str
    total_tokens: float
    phases: list[Phase]
    description: str = ""

    def validate(
        self,
        *,
        max_epochs: float = MAX_EPOCHS,
        atol: float = 1e-6,
        allow_pending_license_review: bool = False,
    ) -> None:
        """Raise :class:`MixtureError` when the recipe cannot safely be trained.

        The check is intentionally strict.  A malformed distribution, an incompatible
        licence, or accidental corpus over-repetition all otherwise produce a perfectly
        plausible-looking training run. ``allow_pending_license_review`` exists only so
        documentation can be generated for a draft recipe; training callers must retain
        its fail-closed default.
        """

        if not math.isfinite(self.total_tokens) or self.total_tokens <= 0:
            raise MixtureError("total_tokens must be a finite positive number")
        if not self.phases:
            raise MixtureError("mixture has no phases")
        if not math.isfinite(max_epochs) or max_epochs <= 0:
            raise ValueError("max_epochs must be a finite positive number")

        self._check_weights(
            [phase.weight for phase in self.phases],
            "phase weights sum",
            atol=atol,
        )

        phase_names: set[str] = set()
        corpus_usage: dict[tuple[str, str, str], tuple[float, float | None, list[str]]] = {}
        for phase in self.phases:
            if not phase.name.strip():
                raise MixtureError("phase name must not be empty")
            if phase.name in phase_names:
                raise MixtureError(f"duplicate phase name: {phase.name}")
            phase_names.add(phase.name)
            if not isinstance(phase.context_len, int) or phase.context_len <= 0:
                raise MixtureError(f"{phase.name}: context_len must be a positive integer")
            if not phase.sources:
                raise MixtureError(f"{phase.name}: phase has no sources")

            self._check_weights(
                [source.weight for source in phase.sources],
                f"{phase.name}: source weights sum",
                atol=atol,
            )

            source_names: set[str] = set()
            phase_tokens = phase.allocated_tokens(self.total_tokens)
            for source in phase.sources:
                label = f"{phase.name}/{source.name}"
                if not source.name.strip() or not source.hf_id.strip() or not source.domain.strip():
                    raise MixtureError(f"{label}: name, hf_id and domain must be non-empty")
                if source.name in source_names:
                    raise MixtureError(f"{phase.name}: duplicate source name {source.name}")
                source_names.add(source.name)

                self._validate_license(
                    source,
                    label,
                    allow_pending_review=allow_pending_license_review,
                )
                if source.available_tokens is not None and (
                    not math.isfinite(source.available_tokens) or source.available_tokens <= 0
                ):
                    raise MixtureError(f"{label}: available_tokens must be positive")
                key = _source_identity(source)
                allocated = source.allocated_tokens(phase_tokens)
                previous = corpus_usage.get(key)
                if previous is None:
                    corpus_usage[key] = (allocated, source.available_tokens, [label])
                else:
                    total, available, labels = previous
                    if (
                        source.available_tokens is not None
                        and available is not None
                        and source.available_tokens != available
                    ):
                        raise MixtureError(
                            f"{label}: available_tokens disagrees with another use of "
                            f"the same corpus ({source.available_tokens:g} != {available:g})"
                        )
                    known_available = (
                        available if available is not None else source.available_tokens
                    )
                    corpus_usage[key] = (
                        total + allocated,
                        known_available,
                        [*labels, label],
                    )

        for allocated, available, labels in corpus_usage.values():
            if available is None:
                continue
            epochs = allocated / available
            if epochs > max_epochs + atol:
                joined = ", ".join(labels)
                raise MixtureError(
                    f"{joined}: cumulative planned {epochs:.2f} epochs exceeds the "
                    f"{max_epochs:g}-epoch limit"
                )

    @staticmethod
    def _check_weights(weights: list[float], label: str, *, atol: float) -> None:
        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            raise MixtureError(f"{label}: weights must be finite and non-negative")
        total = math.fsum(weights)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=atol):
            raise MixtureError(f"{label} to {total:.12g}, expected 1")

    @staticmethod
    def _validate_license(
        source: Source,
        label: str,
        *,
        allow_pending_review: bool = False,
    ) -> None:
        validate_release_license(
            source.license,
            label=label,
            allow_pending_review=allow_pending_review,
        )

    def domain_shares(self) -> dict[str, float]:
        """Return global shares, including each phase's share of the total budget."""

        shares: defaultdict[str, float] = defaultdict(float)
        for phase in self.phases:
            for source in phase.sources:
                shares[source.domain] += phase.weight * source.weight
        return dict(shares)

    def unverified_sources(self) -> list[str]:
        return [
            f"{phase.name}/{source.name}"
            for phase in self.phases
            for source in phase.sources
            if source.available_tokens is None
        ]

    def license_warnings(self) -> list[str]:
        """Return sources blocked pending per-subset or bespoke human review."""

        warnings: list[str] = []
        for phase in self.phases:
            for source in phase.sources:
                key = _license_key(source.license or "")
                if any(marker in key for marker in _REVIEW_LICENSE_KEYS):
                    warnings.append(
                        f"{phase.name}/{source.name}: licence {source.license!r} "
                        "requires REVIEW and cannot train yet"
                    )
        return warnings

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Mixture:
        values = dict(data)
        values["phases"] = [Phase.from_dict(phase) for phase in values.get("phases", [])]
        return cls(**values)

    def to_yaml(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(
                self.to_dict(),
                handle,
                sort_keys=False,
                allow_unicode=True,
                default_flow_style=False,
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> Mixture:
        with Path(path).open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, Mapping):
            raise MixtureError("mixture YAML must contain a mapping")
        return cls.from_dict(data)

    def report(self) -> str:
        """Render the recipe as a compact Markdown model-card section."""

        lines = [
            f"# Data mixture — {self.name}",
            "",
        ]
        if self.description:
            lines.extend((self.description, ""))
        lines.extend((f"Total budget: **{self.total_tokens / 1e9:.1f}B tokens**", ""))

        for phase in self.phases:
            phase_tokens = phase.allocated_tokens(self.total_tokens)
            lines.append(
                f"## Phase {phase.name} — {phase_tokens / 1e9:.1f}B tokens "
                f"({phase.weight:.0%}), context {phase.context_len}, LR {phase.lr_schedule}"
            )
            lines.append("")
            if phase.purpose:
                lines.extend((phase.purpose, ""))
            lines.extend(
                (
                    "| Source | HF id | Domain | Share | Tokens | Epochs | Licence |",
                    "|---|---|---|---:|---:|---:|---|",
                )
            )
            for source in sorted(phase.sources, key=lambda item: item.weight, reverse=True):
                tokens = source.allocated_tokens(phase_tokens)
                epochs = source.epochs(phase_tokens)
                epoch_text = "—" if epochs is None else f"{epochs:.2f}"
                lines.append(
                    f"| {source.name} | `{source.hf_id}` | {source.domain} | "
                    f"{source.weight:.1%} | {tokens / 1e9:.2f}B | {epoch_text} | "
                    f"{source.license} |"
                )
            lines.append("")

        lines.extend(
            (
                "## Aggregate by domain",
                "",
                "| Domain | Tokens | Share |",
                "|---|---:|---:|",
            )
        )
        shares = self.domain_shares()
        for domain, share in sorted(shares.items(), key=lambda item: item[1], reverse=True):
            lines.append(f"| {domain} | {self.total_tokens * share / 1e9:.2f}B | {share:.1%} |")

        counts = Counter(source.license for phase in self.phases for source in phase.sources)
        lines.extend(("", "## Licences", "", "| Licence | Sources |", "|---|---|"))
        for licence, count in sorted(counts.items()):
            lines.append(f"| {licence} | {count} |")

        unverified = self.unverified_sources()
        if unverified:
            lines.extend(
                (
                    "",
                    "## Unverified sizes",
                    "",
                    "Epoch counts could not be checked for these sources; confirm before use:",
                    "",
                )
            )
            lines.extend(f"- {source}" for source in unverified)

        return "\n".join(lines).rstrip() + "\n"
