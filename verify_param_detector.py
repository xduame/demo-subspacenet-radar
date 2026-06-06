"""Verify the radar parameter detector against simulator truth.

For each SNR in ``SNR_LIST`` we run ``NUM_TRIALS`` trials. Each trial:

  1. Builds a fresh ``RadarSamples`` instance and assigns ground-truth DoA.
  2. Generates the array snapshot ``X`` via ``samples_creation``.
  3. Reads in-window truth from ``samples_model._last_true_params``.
  4. Runs ``detect_parameters`` with the truth DoA (this isolates the param
     detector from SubspaceNet's DoA accuracy).
  5. Records per-source (truth, estimate) pairs.

Sources whose chirp band straddles the Nyquist edge (``aliased=True`` in the
truth dict) are reported separately because the synthesized signal in that
case folds around +/- fs/2 and the spectral center loses physical meaning --
the detector's FFT centroid is correct on what's in the data, just not on
what the library "intended". Verify scripts should focus on the non-aliased
slice for headline numbers and surface the aliased slice as a known limit.

Each parameter is summarized by both absolute and relative error medians /
P95 (single-tone BW reports absolute only because the relative denominator
is 0). Sources whose window contained no pulses are skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.param_detector import detect_parameters
from src.radar_samples import RadarSamples
from src.system_model import SystemModelParams


NUM_TRIALS = 100
SNR_LIST = (0, 20)
M_SOURCES = 2
T_SNAPSHOTS = 200
N_SENSORS = 16
FS_MHZ = 200.0
RF_CENTER_MHZ = 9000.0
LIB_PATH = "data/data_manual.xlsx"
MODES_POOL = ["VS", "MRWS", "TASS", "TAST"]
DOA_MIN_GAP_DEG = 15.0  # so beamforming can separate the two sources


@dataclass
class SourceRecord:
    snr_db: int
    doa_deg: float
    truth: dict
    estimate: dict
    aliased: bool
    no_pulse_truth: bool
    no_pulse_detected: bool


def build_params(snr_db: int) -> SystemModelParams:
    return (
        SystemModelParams()
        .set_parameter("N", N_SENSORS)
        .set_parameter("M", M_SOURCES)
        .set_parameter("T", T_SNAPSHOTS)
        .set_parameter("snr", int(snr_db))
        .set_parameter("signal_type", "NarrowBand")
        .set_parameter("signal_nature", "non-coherent")
        .set_parameter("eta", 0)
        .set_parameter("bias", 0)
        .set_parameter("sv_noise_var", 0)
        .set_parameter("data_source", "radar")
        .set_parameter("lib_path", LIB_PATH)
        .set_parameter("fs_mhz", FS_MHZ)
        .set_parameter("rf_center_mhz", RF_CENTER_MHZ)
        .set_parameter("modes_pool", MODES_POOL)
    )


def sample_truth_doas(rng: np.random.Generator) -> list[float]:
    """Draw M DoAs in [-60, 60] with at least DOA_MIN_GAP_DEG separation."""
    while True:
        candidates = rng.uniform(-60.0, 60.0, size=M_SOURCES)
        candidates.sort()
        gaps = np.diff(candidates)
        if np.all(gaps >= DOA_MIN_GAP_DEG):
            return candidates.tolist()


def run_one_trial(snr_db: int, rng_seed: int) -> list[SourceRecord]:
    """One Monte-Carlo trial: returns one record per (truth) source."""
    np.random.seed(rng_seed)            # for the simulator's internal np.random
    rng = np.random.default_rng(rng_seed)

    params = build_params(snr_db)
    samples_model = RadarSamples(params)
    truth_doas = sample_truth_doas(rng)
    samples_model.set_doa(truth_doas)
    X, _, _, _ = samples_model.samples_creation(
        noise_mean=0, noise_variance=1, signal_mean=0, signal_variance=1
    )
    truths = samples_model._last_true_params
    estimates = detect_parameters(
        X,
        truth_doas,
        fs_mhz=FS_MHZ,
        rf_center_mhz=RF_CENTER_MHZ,
    )

    records: list[SourceRecord] = []
    for doa, truth, est in zip(truth_doas, truths, estimates):
        records.append(
            SourceRecord(
                snr_db=snr_db,
                doa_deg=float(doa),
                truth=truth,
                estimate=est.to_dict(),
                aliased=bool(truth.get("aliased", False)),
                no_pulse_truth=int(truth.get("num_pulses_in_window", 0)) == 0,
                no_pulse_detected=int(est.pulse_count) == 0,
            )
        )
    return records


def _percentile(values: Sequence[float], q: float) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def summarize_param(
    records: list[SourceRecord],
    param: str,
    *,
    include_relative: bool,
) -> dict:
    """Return {abs_median, abs_p95, rel_median, rel_p95, n}."""
    abs_errs: list[float] = []
    rel_errs: list[float] = []
    for r in records:
        truth_val = r.truth.get(param)
        est_val = r.estimate.get(param)
        if truth_val is None or est_val is None:
            continue
        if not np.isfinite(truth_val) or not np.isfinite(est_val):
            continue
        abs_err = abs(est_val - truth_val)
        abs_errs.append(abs_err)
        if include_relative and abs(truth_val) > 1e-9:
            rel_errs.append(abs_err / abs(truth_val))
    return dict(
        abs_median=_percentile(abs_errs, 50),
        abs_p95=_percentile(abs_errs, 95),
        rel_median=_percentile(rel_errs, 50) if include_relative else float("nan"),
        rel_p95=_percentile(rel_errs, 95) if include_relative else float("nan"),
        n=len(abs_errs),
    )


def print_summary(
    title: str,
    records: list[SourceRecord],
    include_relative_bw: bool,
) -> None:
    """Pretty-print one parameter table."""
    print(f"\n--- {title} (n={len(records)} sources) ---")
    if not records:
        print("  (no sources to summarize)")
        return

    rows = [
        ("PW (us) ", "pw_us", True),
        ("RF (MHz)", "rf_mhz", True),
        ("BW (MHz)", "bw_mhz", include_relative_bw),
        ("TOA(us) ", "toa_us", False),
    ]
    print(
        f"  {'param':<10} {'abs.median':>11} {'abs.P95':>11} "
        f"{'rel.median':>11} {'rel.P95':>10}   n"
    )
    for label, key, rel in rows:
        s = summarize_param(records, key, include_relative=rel)
        rel_med = f"{100 * s['rel_median']:.2f}%" if rel and np.isfinite(s["rel_median"]) else "  --   "
        rel_p95 = f"{100 * s['rel_p95']:.2f}%" if rel and np.isfinite(s["rel_p95"]) else "  --   "
        abs_med = f"{s['abs_median']:.4g}" if np.isfinite(s["abs_median"]) else "nan"
        abs_p95 = f"{s['abs_p95']:.4g}" if np.isfinite(s["abs_p95"]) else "nan"
        print(
            f"  {label:<10} {abs_med:>11} {abs_p95:>11} "
            f"{rel_med:>11} {rel_p95:>10}   {s['n']}"
        )


def main() -> None:
    print("Radar Parameter Detector Verification")
    print("=" * 70)
    print(f"Trials per SNR : {NUM_TRIALS}")
    print(f"SNRs           : {SNR_LIST} dB")
    print(f"M={M_SOURCES}, N={N_SENSORS}, T={T_SNAPSHOTS}, fs={FS_MHZ} MHz, "
          f"rf_center={RF_CENTER_MHZ} MHz")
    print(f"Modes pool     : {MODES_POOL}")
    print()

    all_records: list[SourceRecord] = []
    for snr_db in SNR_LIST:
        for trial_idx in range(NUM_TRIALS):
            # Distinct seed per (snr, trial) so the two SNRs see different
            # realizations -- comparing identical signals at different SNRs
            # would only test the noise path, not the algorithm.
            seed = 10_000 * (snr_db + 100) + trial_idx
            all_records.extend(run_one_trial(snr_db, seed))

    for snr_db in SNR_LIST:
        snr_records = [r for r in all_records if r.snr_db == snr_db]
        valid = [
            r for r in snr_records
            if not r.no_pulse_truth and not r.no_pulse_detected
        ]
        non_aliased = [r for r in valid if not r.aliased]
        aliased = [r for r in valid if r.aliased]
        no_pulse_truth = sum(r.no_pulse_truth for r in snr_records)
        no_pulse_det = sum(
            r.no_pulse_detected and not r.no_pulse_truth for r in snr_records
        )

        print()
        print("=" * 70)
        print(f"SNR = {snr_db:+d} dB")
        print("=" * 70)
        print(f"  Total source-trials   : {len(snr_records)}")
        print(f"  Skipped (no in-window truth)   : {no_pulse_truth}")
        print(f"  Skipped (no pulse detected)    : {no_pulse_det}")
        print(f"  Non-aliased (clean) sources    : {len(non_aliased)}")
        print(f"  Aliased sources (chirp wraps Nyquist) : {len(aliased)}")

        print_summary(
            "Non-aliased (use these for headline accuracy)",
            non_aliased,
            include_relative_bw=True,
        )
        print_summary(
            "Aliased (chirp band exceeds Nyquist; informational only)",
            aliased,
            include_relative_bw=False,
        )

    print()
    print("Notes:")
    print(" - BW relative error is suppressed on aliased sources because the")
    print("   chirp folds around +/- fs/2, making the spectral 'width' physically")
    print("   ambiguous. If the headline non-aliased set is empty, the library")
    print("   modes used (modes_pool) all overflow Nyquist for rf_center_mhz=")
    print(f"   {RF_CENTER_MHZ}; raise rf_center_mhz or restrict modes_pool to")
    print("   modes whose RF band fits in (rf_center - fs/2, rf_center + fs/2).")
    print(" - TOA reports absolute error only (truth=0 for the first in-window")
    print("   pulse makes relative meaningless).")


if __name__ == "__main__":
    main()
