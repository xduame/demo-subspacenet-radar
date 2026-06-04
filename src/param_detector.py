"""Beamforming-based radar pulse parameter detection.

This module is intentionally independent from SubspaceNet. It expects an array
snapshot matrix ``X`` and DoA estimates, beamforms each DoA into a single
complex channel, then measures pulse parameters from the separated waveform.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class PulseEstimate:
    """Parameter estimates for one detected pulse."""

    toa_us: float
    pw_us: float
    rf_mhz: float
    bw_mhz: float
    start_index: int
    end_index: int


@dataclass
class SourceParamEstimate:
    """Pulse-parameter summary for one beamformed source."""

    doa_deg: float
    pulse_count: int
    toa_us: float
    pw_us: float
    rf_mhz: float
    bw_mhz: float
    pulses: list[PulseEstimate]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["pulses"] = [asdict(pulse) for pulse in self.pulses]
        return data


def steering_vector(
    doa_deg: float, n_sensors: int = 16, element_spacing: float = 0.5
) -> np.ndarray:
    """Return the ULA steering vector used by the narrowband simulator."""
    sensor_index = np.arange(n_sensors)
    theta = np.deg2rad(doa_deg)
    return np.exp(-2j * np.pi * element_spacing * sensor_index * np.sin(theta))


def beamform(X, doa_deg, N=16):
    """
    用给定 DoA 对阵列信号做相干合成,分离出该方向的源信号。

    Args:
        X: 阵列信号 [N, T] 复数 (numpy 或 torch)
        doa_deg: 单个角度(度)
        N: 阵元数

    Returns:
        beamformed: [T] 复数,该方向波束形成后的单通道信号
    """
    if hasattr(X, "detach"):
        X = X.detach().cpu().numpy()
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"Expected X with shape (N, T), got {X.shape}.")
    if X.shape[0] != N:
        raise ValueError(f"Expected {N} sensors, got {X.shape[0]}.")
    if not np.iscomplexobj(X):
        raise ValueError("Expected complex array samples in X.")

    n = np.arange(N)
    theta = np.deg2rad(float(doa_deg))
    steering = np.exp(-1j * np.pi * n * np.sin(theta))
    return np.conj(steering) @ X / N


def beamform_source(
    X: np.ndarray, doa_deg: float, element_spacing: float = 0.5
) -> np.ndarray:
    """Delay-and-sum beamform an array signal toward one DoA."""
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"Expected X with shape (N, T), got {X.shape}.")
    if not np.iscomplexobj(X):
        raise ValueError("Expected complex array samples in X.")

    if element_spacing == 0.5:
        return beamform(X, doa_deg=doa_deg, N=X.shape[0])

    steering = steering_vector(
        doa_deg, n_sensors=X.shape[0], element_spacing=element_spacing
    )
    return np.conj(steering) @ X / X.shape[0]


def beamform_sources(
    X: np.ndarray, doa_deg: np.ndarray | list[float], element_spacing: float = 0.5
) -> list[np.ndarray]:
    """Beamform the array observation into one channel per DoA."""
    return [
        beamform_source(X=X, doa_deg=float(doa), element_spacing=element_spacing)
        for doa in np.asarray(doa_deg, dtype=float)
    ]


def detect_toa_pw(sig, fs_mhz):
    """
    双门限法检测脉冲的 TOA 和 PW。

    Args:
        sig: [T] 复数信号(波束形成后的单源信号)
        fs_mhz: 采样率(MHz)

    Returns:
        toa_us: 到达时间(微秒)
        pw_us: 脉宽(微秒)
    """
    if fs_mhz <= 0:
        raise ValueError("fs_mhz must be positive.")
    if hasattr(sig, "detach"):
        sig = sig.detach().cpu().numpy()

    sig = np.asarray(sig)
    if sig.ndim != 1:
        raise ValueError(f"Expected sig with shape (T,), got {sig.shape}.")
    if sig.size == 0:
        return float("nan"), float("nan")

    env = np.abs(sig).astype(float)
    env[~np.isfinite(env)] = 0.0
    peak = float(env.max())
    if peak <= 0:
        return float("nan"), float("nan")
    if np.ptp(env) <= max(np.finfo(float).eps, 1e-12 * peak):
        return float("nan"), float("nan")

    high_mask = env > 0.7 * peak
    high_edges = np.diff(np.concatenate([[False], high_mask, [False]]).astype(int))
    high_starts = np.flatnonzero(high_edges == 1)
    high_ends = np.flatnonzero(high_edges == -1)
    has_body = any(end - start >= 2 for start, end in zip(high_starts, high_ends))
    if not has_body:
        return float("nan"), float("nan")

    c1 = np.where(high_mask)[0]
    if c1.size < 2:
        return float("nan"), float("nan")
    th_high = float(np.mean(env[c1]))
    c10 = np.where(env >= th_high * (1 - 1e-9))[0]
    if c10.size == 0:
        c10 = c1
    toa_idx = int(c10[0])

    c2 = np.where(env > 0.2 * peak)[0]
    if c2.size == 0:
        return float("nan"), float("nan")
    th_low = float(np.mean(env[c2]))

    tail = env[toa_idx:]
    below = np.where(tail < th_low * (1 - 1e-9))[0]
    below = below[below > 0]
    eoa_idx = int(toa_idx + below[0]) if below.size else int(len(env) - 1)
    if eoa_idx <= toa_idx:
        return float("nan"), float("nan")

    toa_us = toa_idx / fs_mhz
    pw_us = (eoa_idx - toa_idx) / fs_mhz
    return float(toa_us), float(pw_us)


def _active_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate([[False], mask, [False]]).astype(int))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def _merge_short_gaps(
    segments: list[tuple[int, int]], max_gap_samples: int
) -> list[tuple[int, int]]:
    if not segments:
        return []

    merged = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= max_gap_samples:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def detect_pulse_segments(
    signal: np.ndarray,
    threshold_rel: float = 0.35,
    noise_percentile: float = 20.0,
    min_width_samples: int = 2,
    max_gap_samples: int = 1,
) -> list[tuple[int, int]]:
    """Detect pulse support intervals from a beamformed complex signal."""
    signal = np.asarray(signal)
    envelope = np.abs(signal)
    if envelope.size == 0 or np.max(envelope) <= 0:
        return []

    floor = np.percentile(envelope, noise_percentile)
    peak = np.max(envelope)
    threshold = floor + threshold_rel * (peak - floor)
    segments = _active_segments(envelope >= threshold)
    segments = _merge_short_gaps(segments, max_gap_samples=max_gap_samples)
    return [
        (start, end)
        for start, end in segments
        if end - start >= min_width_samples
    ]


def instantaneous_frequency_mhz(signal: np.ndarray, fs_mhz: float) -> np.ndarray:
    """Estimate sample-to-sample instantaneous frequency in MHz."""
    signal = np.asarray(signal)
    if signal.size < 2:
        return np.array([], dtype=float)
    phase = np.unwrap(np.angle(signal))
    return np.diff(phase) * fs_mhz / (2 * np.pi)


def _finite_median(values: list[float]) -> float:
    values_array = np.asarray(values, dtype=float)
    values_array = values_array[np.isfinite(values_array)]
    if values_array.size == 0:
        return float("nan")
    return float(np.median(values_array))


def _estimate_pulse(
    start: int,
    end: int,
    inst_freq_mhz: np.ndarray,
    fs_mhz: float,
    rf_center_mhz: float,
) -> PulseEstimate:
    freq_slice = inst_freq_mhz[start : max(start, end - 1)]
    if freq_slice.size:
        low, high = np.percentile(freq_slice, [5, 95])
        rf_mhz = rf_center_mhz + float(np.median(freq_slice))
        bw_mhz = float(abs(high - low))
    else:
        rf_mhz = float("nan")
        bw_mhz = float("nan")

    return PulseEstimate(
        toa_us=float(start / fs_mhz),
        pw_us=float((end - start) / fs_mhz),
        rf_mhz=rf_mhz,
        bw_mhz=bw_mhz,
        start_index=int(start),
        end_index=int(end),
    )


def estimate_source_params(
    signal: np.ndarray,
    doa_deg: float,
    fs_mhz: float = 200.0,
    rf_center_mhz: float = 9000.0,
    threshold_rel: float = 0.35,
    min_width_samples: int = 2,
) -> SourceParamEstimate:
    """Estimate pulse parameters from one beamformed source channel."""
    segments = detect_pulse_segments(
        signal,
        threshold_rel=threshold_rel,
        min_width_samples=min_width_samples,
    )
    inst_freq = instantaneous_frequency_mhz(signal, fs_mhz=fs_mhz)
    pulses = [
        _estimate_pulse(start, end, inst_freq, fs_mhz, rf_center_mhz)
        for start, end in segments
    ]

    if not pulses:
        return SourceParamEstimate(
            doa_deg=float(doa_deg),
            pulse_count=0,
            toa_us=float("nan"),
            pw_us=float("nan"),
            rf_mhz=float("nan"),
            bw_mhz=float("nan"),
            pulses=[],
        )

    return SourceParamEstimate(
        doa_deg=float(doa_deg),
        pulse_count=len(pulses),
        toa_us=pulses[0].toa_us,
        pw_us=_finite_median([pulse.pw_us for pulse in pulses]),
        rf_mhz=_finite_median([pulse.rf_mhz for pulse in pulses]),
        bw_mhz=_finite_median([pulse.bw_mhz for pulse in pulses]),
        pulses=pulses,
    )


def detect_parameters(
    X: np.ndarray,
    doa_deg: np.ndarray | list[float],
    fs_mhz: float = 200.0,
    rf_center_mhz: float = 9000.0,
    element_spacing: float = 0.5,
    threshold_rel: float = 0.35,
    min_width_samples: int = 2,
) -> list[SourceParamEstimate]:
    """Beamform each DoA and estimate pulse parameters per source."""
    estimates = []
    for doa, signal in zip(
        np.asarray(doa_deg, dtype=float),
        beamform_sources(X, doa_deg, element_spacing=element_spacing),
    ):
        estimates.append(
            estimate_source_params(
                signal=signal,
                doa_deg=float(doa),
                fs_mhz=fs_mhz,
                rf_center_mhz=rf_center_mhz,
                threshold_rel=threshold_rel,
                min_width_samples=min_width_samples,
            )
        )
    return estimates


def detect_parameters_as_dict(*args, **kwargs) -> list[dict]:
    """Return ``detect_parameters`` results as plain dictionaries."""
    return [estimate.to_dict() for estimate in detect_parameters(*args, **kwargs)]
