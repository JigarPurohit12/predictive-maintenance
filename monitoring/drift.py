"""Population Stability Index per feature, against the training distribution.

    python -m monitoring.drift --build-reference     # after training
    python -m monitoring.drift --batch scores/dt=2015-11-01

A model does not announce that it has stopped working. The sensors get
recalibrated, a batch of machines gets replaced, a firmware update changes what
"normal vibration" means — and the model keeps returning confident probabilities
about a world that no longer exists. PSI is the cheapest thing that notices.

    PSI = sum over bins of (actual% - expected%) x ln(actual% / expected%)

Read it as: how far has this feature's distribution moved since training. The
conventional bands are < 0.1 stable, 0.1-0.2 moderate, > 0.2 significant, and
they are conventions rather than laws — what matters is that the number is
tracked over time, because a feature climbing steadily from 0.02 to 0.18 is
telling you something well before it crosses anyone's line.

**PSI does not measure whether the model is still accurate.** It measures
whether the inputs still look like the ones it was trained on. A feature can
drift a long way without hurting performance, and performance can collapse with
no drift at all. That is what ``monitoring/performance.py`` is for; the two
together are the picture, and neither alone is.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from config import Settings, configure_logging, get_settings
from features.contract import FeatureContract, build_contract

LOGGER = logging.getLogger(__name__)

REFERENCE_FILENAME = "drift_reference.json"
DRIFT_REPORT_FILENAME = "drift_report.json"

#: Conventional PSI bands. Not laws — see the module docstring.
PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.20

DEFAULT_BINS = 10

#: Floor applied to a bin's proportion before taking a logarithm. Without it a
#: bin that is empty in one distribution and populated in the other sends PSI to
#: infinity, which is not more informative than "very large".
EPSILON = 1e-6


class FeatureDrift(BaseModel):
    """PSI for one feature."""

    model_config = ConfigDict(frozen=True)

    feature: str
    psi: float
    severity: str
    reference_mean: float
    batch_mean: float

    @property
    def significant(self) -> bool:
        return self.psi >= PSI_SIGNIFICANT


class ReferenceProfile(BaseModel):
    """Binned training distribution, per feature.

    Saved next to the model so a scoring run can compare against it without
    loading the training data — which on AWS would mean a Glue job reading the
    whole feature matrix to score two hundred rows.
    """

    model_config = ConfigDict(frozen=True)

    feature_hash: str
    built_at: datetime
    rows: int
    n_bins: int
    #: feature -> the *interior* cut points, always finite and possibly empty.
    #: The outermost bins are open by construction; storing them as infinities
    #: would not survive the round trip, because JSON has no infinity literal
    #: and pydantic serialises one as null.
    edges: dict[str, list[float]]
    #: feature -> proportion of reference rows in each bin
    proportions: dict[str, list[float]]
    means: dict[str, float]


class DriftReport(BaseModel):
    """One batch, compared against the reference."""

    model_config = ConfigDict(frozen=True)

    computed_at: datetime
    rows: int
    feature_hash: str
    features: tuple[FeatureDrift, ...]

    @property
    def drifted(self) -> tuple[FeatureDrift, ...]:
        return tuple(feature for feature in self.features if feature.significant)

    @property
    def worst(self) -> FeatureDrift | None:
        return max(self.features, key=lambda feature: feature.psi) if self.features else None

    @property
    def ok(self) -> bool:
        return not self.drifted


def severity_for(psi: float) -> str:
    if psi >= PSI_SIGNIFICANT:
        return "significant"
    if psi >= PSI_MODERATE:
        return "moderate"
    return "stable"


def _cut_points(values: np.ndarray, n_bins: int) -> list[float]:
    """Interior quantile cut points. Finite, sorted, possibly empty.

    Quantiles rather than equal width: an equal-width binning of a skewed
    feature puts 99% of the reference rows in one bin, and PSI computed against
    that is noise. Duplicate quantiles are collapsed, so a near-constant feature
    ends up with fewer bins rather than with zero-width ones.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return []
    quantiles = np.quantile(finite, np.linspace(0.0, 1.0, n_bins + 1))
    return np.unique(quantiles[1:-1]).tolist()


def _bins(cut_points: list[float]) -> list[float]:
    """Cut points to histogram bins, open at both ends."""
    return [-np.inf, *cut_points, np.inf]


def _proportions(values: np.ndarray, cut_points: list[float]) -> np.ndarray:
    counts, _ = np.histogram(values[np.isfinite(values)], bins=_bins(cut_points))
    total = counts.sum()
    if total == 0:
        return np.full(len(counts), 1.0 / len(counts))
    return counts / total


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    """PSI between two binned distributions.

    Symmetric in the sense that swapping the arguments gives the same number,
    which is worth knowing when reading one: it says "these differ by this
    much", not "this one moved in that direction".
    """
    expected = np.clip(np.asarray(expected, dtype="float64"), EPSILON, None)
    actual = np.clip(np.asarray(actual, dtype="float64"), EPSILON, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def build_reference(
    frame: pd.DataFrame,
    contract: FeatureContract,
    *,
    n_bins: int = DEFAULT_BINS,
    built_at: datetime | None = None,
) -> ReferenceProfile:
    """Bin the training distribution, once, after training.

    Build it from the **training split only**. Building it from the whole
    matrix means the reference already contains the validation and test periods,
    so drift into those periods is invisible by construction.
    """
    edges: dict[str, list[float]] = {}
    proportions: dict[str, list[float]] = {}
    means: dict[str, float] = {}

    for feature in contract.order:
        values = frame[feature].to_numpy(dtype="float64")
        cut_points = _cut_points(values, n_bins)
        edges[feature] = cut_points
        proportions[feature] = _proportions(values, cut_points).tolist()
        means[feature] = float(np.nanmean(values)) if values.size else 0.0

    return ReferenceProfile(
        feature_hash=contract.hash,
        built_at=built_at or datetime.now(),
        rows=len(frame),
        n_bins=n_bins,
        edges=edges,
        proportions=proportions,
        means=means,
    )


def compute_drift(
    profile: ReferenceProfile,
    frame: pd.DataFrame,
    *,
    computed_at: datetime | None = None,
) -> DriftReport:
    """PSI for every feature in the profile, against this batch."""
    missing = [feature for feature in profile.edges if feature not in frame.columns]
    if missing:
        raise ValueError(f"The batch is missing {len(missing)} profiled feature(s): {missing[:5]}")

    drifts: list[FeatureDrift] = []
    for feature, cut_points in profile.edges.items():
        values = frame[feature].to_numpy(dtype="float64")
        actual = _proportions(values, cut_points)
        score = psi(np.array(profile.proportions[feature]), actual)
        drifts.append(
            FeatureDrift(
                feature=feature,
                psi=score,
                severity=severity_for(score),
                reference_mean=profile.means[feature],
                batch_mean=float(np.nanmean(values)) if values.size else 0.0,
            )
        )

    drifts.sort(key=lambda drift: drift.psi, reverse=True)
    return DriftReport(
        computed_at=computed_at or datetime.now(),
        rows=len(frame),
        feature_hash=profile.feature_hash,
        features=tuple(drifts),
    )


def format_report(report: DriftReport, top_n: int = 15) -> str:
    if not report.features:
        return "(no features profiled)"
    frame = pd.DataFrame([feature.model_dump() for feature in report.features]).head(top_n)
    frame["psi"] = frame["psi"].map("{:.4f}".format)
    lines = [
        f"Drift report — {report.rows:,} rows, contract {report.feature_hash}, {report.computed_at:%Y-%m-%d %H:%M}",
        frame.to_string(index=False),
    ]
    if report.drifted:
        names = ", ".join(feature.feature for feature in report.drifted[:5])
        lines.append(f"\n{len(report.drifted)} feature(s) above PSI {PSI_SIGNIFICANT}: {names}")
        lines.append("PSI says the inputs moved, not that the model got worse. Check performance.py too.")
    else:
        lines.append(f"\nNothing above PSI {PSI_SIGNIFICANT}.")
    return "\n".join(lines)


def save_reference(profile: ReferenceProfile, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    LOGGER.info("Wrote %s", path)
    return path


def load_reference(path: Path) -> ReferenceProfile:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Run `python -m monitoring.drift --build-reference`.")
    return ReferenceProfile.model_validate_json(path.read_text(encoding="utf-8"))


def reference_path(settings: Settings) -> Path:
    return settings.features_dir / REFERENCE_FILENAME


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build-reference", action="store_true", help="bin the training split and save it")
    parser.add_argument("--batch", type=Path, default=None, help="parquet of features to check")
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS)
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings)
    contract = build_contract(settings)

    if args.build_reference:
        from features.build_features import load_matrix

        matrix = load_matrix(settings)
        training = matrix[matrix["split"] == "train"]
        profile = build_reference(training, contract, n_bins=args.bins)
        save_reference(profile, reference_path(settings))
        print(f"Reference built from {profile.rows:,} training rows, {len(profile.edges)} features.")
        return 0

    if args.batch is None:
        parser.error("pass --build-reference or --batch")

    profile = load_reference(reference_path(settings))
    report = compute_drift(profile, pd.read_parquet(args.batch))
    print(format_report(report))

    output: dict[str, Any] = json.loads(report.model_dump_json())
    Path("results").mkdir(parents=True, exist_ok=True)
    (Path("results") / DRIFT_REPORT_FILENAME).write_text(json.dumps(output, indent=2), encoding="utf-8")
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
