# 46_stage_r7a_pi5_cached_enrollment_benchmark.py
#
# R7-A — Cached-Enrollment Raspberry Pi 5 Benchmark
#
# PURPOSE
# -------
# Reviewer-requested deployment-realistic latency benchmark in which
# enrollment embeddings are precomputed/cached BEFORE timed verification.
#
# IMPORTANT
# ---------
# This protocol is NOT directly timing-equivalent to Stage 10A.
#
# Stage 10A:
#   enrollment + test audio processing and embedding inference were timed.
#
# R7-A:
#   enrollment embeddings for BOTH b0 and b6 are cached before benchmark.
#   timed region contains:
#       test audio loading/preprocessing
#       + test embedding inference
#       + cosine scoring
#       + routing
#       + b6 test inference if required.
#
# No test embedding caching is allowed.
# Frozen-B10 policy is NOT retuned.

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import psutil
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly
from sklearn.metrics import roc_auc_score, roc_curve


# ============================================================
# CONFIG
# ============================================================

@dataclass(frozen=True)
class Config:
    # Canonical ReDimNet paths from Stage 10A
    redimnet_source: Path = Path(
        "external/ReDimNet"
    )

    b0_checkpoint: Path = Path(
        "models/redimnet/b0-vox2-ptn.pt"
    )

    b6_checkpoint: Path = Path(
        "models/redimnet/b6-vox2-ptn.pt"
    )

    project_root: Path = Path(
        "."
    )

    # If None, script searches canonical Stage09C/Stage10A locations.
    explicit_manifest: str | None = None

    output_dir: Path = Path(
        "./"
        "results/reviewer_audit/"
        "r7a_cached_enrollment_pi5_benchmark"
    )

    sample_rate: int = 16000

    # Frozen-B10 policy — immutable
    tau_b0: float = 0.336179674
    delta_b10: float = 0.134631842

    # Canonical Stage09C / Stage10A subset
    expected_trials: int = 500
    expected_target: int = 250
    expected_impostor: int = 250

    expected_manifest_sha256: str = (
        "d4e26be792f1ebcb2bf83719d1ee303399cde0dbb299248b6258b4ad436aa2c0"
    )

    # Pi 5 canonical CPU configuration
    torch_threads: int = 4
    torch_interop_threads: int = 1

    warmup_files_per_model: int = 5


CFG = Config()


# ============================================================
# GENERAL HELPERS
# ============================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def percentile(values: List[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def current_rss_mb() -> float:
    process = psutil.Process(os.getpid())
    return float(process.memory_info().rss / (1024.0 ** 2))


# ============================================================
# AUDIO
# Canonical Stage 10A preprocessing
# ============================================================

def audio(path: Path, target_sr: int) -> Tuple[torch.Tensor, float]:
    x, sr = sf.read(
        str(path),
        dtype="float32",
        always_2d=False,
    )

    if x.ndim == 2:
        x = np.mean(x, axis=1)

    x = np.asarray(x, dtype=np.float32)

    if int(sr) != int(target_sr):
        g = math.gcd(int(sr), int(target_sr))

        up = int(target_sr) // g
        down = int(sr) // g

        x = resample_poly(x, up, down).astype(
            np.float32,
            copy=False,
        )

    duration_s = float(len(x) / target_sr)

    w = torch.from_numpy(
        np.ascontiguousarray(x)
    ).float().unsqueeze(0)

    return w, duration_s


# ============================================================
# MODEL
# Canonical Stage 10A loader
# ============================================================

def load_model(checkpoint: Path):
    if str(CFG.redimnet_source) not in sys.path:
        sys.path.insert(0, str(CFG.redimnet_source))

    from redimnet.model import ReDimNetWrap

    full = torch.load(
        str(checkpoint),
        map_location="cpu",
        weights_only=False,
    )

    if "model_config" not in full:
        raise RuntimeError(
            f"model_config missing in checkpoint: {checkpoint}"
        )

    if "state_dict" not in full:
        raise RuntimeError(
            f"state_dict missing in checkpoint: {checkpoint}"
        )

    m = ReDimNetWrap(**full["model_config"])

    m.load_state_dict(
        full["state_dict"],
        strict=True,
    )

    m.eval()
    m.cpu()

    return m


@torch.inference_mode()
def emb(model, w: torch.Tensor) -> torch.Tensor:
    z = model(w)

    if isinstance(z, (tuple, list)):
        z = z[0]

    if isinstance(z, dict):
        # Defensive compatibility only.
        # Canonical checkpoints normally do not require this branch.
        for key in ("embedding", "embeddings", "x", "output"):
            if key in z:
                z = z[key]
                break
        else:
            raise RuntimeError(
                f"Unsupported model output dict keys: {list(z.keys())}"
            )

    if z.ndim == 3:
        z = z.mean(dim=1)

    if z.ndim == 1:
        z = z.unsqueeze(0)

    z = F.normalize(
        z,
        p=2,
        dim=1,
    )

    return z.cpu()


def cosine_score(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(
            a,
            b,
            dim=1,
        ).item()
    )


# ============================================================
# MANIFEST DISCOVERY
# ============================================================

def candidate_manifest_paths() -> List[Path]:
    candidates: List[Path] = []

    if CFG.explicit_manifest is not None:
        candidates.append(Path(CFG.explicit_manifest))

    roots = [
        CFG.project_root / "results/stage09c_frozen_b10_sequential",
        CFG.project_root / "results/stage09c_pi5_frozen_b10_sequential",
        CFG.project_root / "results/stage10a_pi5_direct_pipeline_benchmark",
        CFG.project_root / "results/stage09b_pi5_benchmark",
        CFG.project_root / "results",
    ]

    names = [
        "stage09c_manifest.csv",
        "stage09c_500_trials.csv",
        "trial_manifest.csv",
        "trials.csv",
        "manifest.csv",
    ]

    for root in roots:
        for name in names:
            candidates.append(root / name)

    return candidates


def discover_manifest() -> Path:
    # First try known candidate names.
    for p in candidate_manifest_paths():
        if p.exists() and p.is_file():
            if sha256_file(p) == CFG.expected_manifest_sha256:
                return p

    # Then search project results recursively by exact SHA256.
    results_root = CFG.project_root / "results"

    if results_root.exists():
        for p in results_root.rglob("*.csv"):
            try:
                if sha256_file(p) == CFG.expected_manifest_sha256:
                    return p
            except OSError:
                continue

    raise FileNotFoundError(
        "\nCould not locate canonical 500-trial manifest.\n"
        f"Expected SHA256:\n{CFG.expected_manifest_sha256}\n\n"
        "Set Config.explicit_manifest to the exact Stage09C/Stage10A "
        "manifest path if necessary."
    )


# ============================================================
# MANIFEST PARSING
# ============================================================

def normalize_label(value: str) -> int:
    s = str(value).strip().lower()

    if s in {
        "1",
        "target",
        "same",
        "positive",
        "pos",
        "true",
    }:
        return 1

    if s in {
        "0",
        "nontarget",
        "non-target",
        "different",
        "impostor",
        "negative",
        "neg",
        "false",
    }:
        return 0

    # Some verification files use -1 for impostor.
    if s == "-1":
        return 0

    raise ValueError(f"Unknown trial label: {value}")


def first_existing(
    row: Dict[str, str],
    keys: Tuple[str, ...],
) -> str | None:
    for key in keys:
        if key in row:
            value = row[key]

            if value is not None and str(value).strip():
                return str(value).strip()

    return None


def resolve_audio_path(raw: str) -> Path:
    p = Path(raw)

    if p.is_absolute():
        return p

    # Canonical Raspberry Pi 5 VoxCeleb1-O root
    audio_root = (
        CFG.project_root
        / "DATASETS"
        / "VoxCeleb1_O"
    )

    q = audio_root / p

    if q.exists():
        return q

    raise FileNotFoundError(
        f"Audio file not found under canonical VoxCeleb1-O rooexternal/n{q}"
    )


def read_manifest(path: Path) -> List[dict]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise RuntimeError("Manifest has no CSV header.")

        rows = []

        for idx, row in enumerate(reader):
            enroll_raw = first_existing(
                row,
                (
                    "enroll_rel",       # canonical Stage09C/10A manifest
                    "enroll_path",
                    "enroll",
                    "enroll_wav",
                    "enroll_file",
                    "path1",
                    "wav1",
                    "file1",
                    "audio1",
                    "utt1",
                ),
            )

            test_raw = first_existing(
                row,
                (
                    "test_rel",         # canonical Stage09C/10A manifest
                    "test_path",
                    "test",
                    "test_wav",
                    "test_file",
                    "path2",
                    "wav2",
                    "file2",
                    "audio2",
                    "utt2",
                ),
            )

            label_raw = first_existing(
                row,
                (
                    "label",
                    "target",
                    "y",
                    "same",
                    "is_target",
                ),
            )

            if enroll_raw is None:
                raise RuntimeError(
                    f"Cannot find enrollment path column.\n"
                    f"Columns: {reader.fieldnames}"
                )

            if test_raw is None:
                raise RuntimeError(
                    f"Cannot find test path column.\n"
                    f"Columns: {reader.fieldnames}"
                )

            if label_raw is None:
                raise RuntimeError(
                    f"Cannot find label column.\n"
                    f"Columns: {reader.fieldnames}"
                )

            enroll_path = resolve_audio_path(enroll_raw)
            test_path = resolve_audio_path(test_raw)

            if not enroll_path.exists():
                raise FileNotFoundError(
                    f"Enrollment audio missinexternal/n{enroll_path}"
                )

            if not test_path.exists():
                raise FileNotFoundError(
                    f"Test audio missinexternal/n{test_path}"
                )

            rows.append(
                {
                    "trial_index": idx,
                    "label": normalize_label(label_raw),
                    "enroll_path": enroll_path,
                    "test_path": test_path,
                }
            )

    return rows


# ============================================================
# VERIFICATION METRICS
# Same convention as project
# ============================================================

def compute_eer(
    labels: np.ndarray,
    scores: np.ndarray,
) -> Tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(
        labels,
        scores,
        pos_label=1,
    )

    fnr = 1.0 - tpr

    idx = int(
        np.nanargmin(
            np.abs(fnr - fpr)
        )
    )

    eer = (
        float(fpr[idx]) + float(fnr[idx])
    ) / 2.0

    return (
        100.0 * eer,
        float(thresholds[idx]),
    )


def compute_min_dcf(
    labels: np.ndarray,
    scores: np.ndarray,
) -> Tuple[float, float]:
    p_target = 0.01
    c_miss = 1.0
    c_fa = 1.0

    fpr, tpr, thresholds = roc_curve(
        labels,
        scores,
        pos_label=1,
    )

    fnr = 1.0 - tpr

    c_det = (
        c_miss * fnr * p_target
        + c_fa * fpr * (1.0 - p_target)
    )

    c_default = min(
        c_miss * p_target,
        c_fa * (1.0 - p_target),
    )

    idx = int(np.argmin(c_det))

    return (
        float(c_det[idx] / c_default),
        float(thresholds[idx]),
    )


def verification_metrics(
    labels: List[int],
    scores: List[float],
) -> dict:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)

    eer, eer_thr = compute_eer(y, s)
    min_dcf, dcf_thr = compute_min_dcf(y, s)
    auc = float(roc_auc_score(y, s))

    return {
        "eer_percent": eer,
        "eer_threshold": eer_thr,
        "min_dcf": min_dcf,
        "min_dcf_threshold": dcf_thr,
        "auc": auc,
    }


# ============================================================
# LATENCY SUMMARY
# ============================================================

def latency_summary(
    latencies_ms: List[float],
    test_durations_s: List[float],
) -> dict:
    x = np.asarray(
        latencies_ms,
        dtype=np.float64,
    )

    d = np.asarray(
        test_durations_s,
        dtype=np.float64,
    )

    rtf = (x / 1000.0) / d

    return {
        "mean_latency_ms": float(np.mean(x)),
        "median_latency_ms": float(np.median(x)),
        "p95_latency_ms": float(np.percentile(x, 95)),
        "p99_latency_ms": float(np.percentile(x, 99)),
        "min_latency_ms": float(np.min(x)),
        "max_latency_ms": float(np.max(x)),
        "mean_test_rtf": float(np.mean(rtf)),
    }


# ============================================================
# WARMUP
# ============================================================

def warmup_models(
    model_b0,
    model_b6,
    trials: List[dict],
):
    unique_files: List[Path] = []
    seen = set()

    for row in trials:
        for p in (
            row["enroll_path"],
            row["test_path"],
        ):
            key = str(p)

            if key not in seen:
                seen.add(key)
                unique_files.append(p)

            if len(unique_files) >= CFG.warmup_files_per_model:
                break

        if len(unique_files) >= CFG.warmup_files_per_model:
            break

    print(
        f"\nWarm-up using {len(unique_files)} unique audio files..."
    )

    for model_name, model_obj in (
        ("b0", model_b0),
        ("b6", model_b6),
    ):
        t0 = time.perf_counter()

        for p in unique_files:
            w, _ = audio(p, CFG.sample_rate)
            _ = emb(model_obj, w)

        elapsed = time.perf_counter() - t0

        print(
            f"  {model_name}: warm-up complete "
            f"({elapsed:.2f} s)"
        )


# ============================================================
# ENROLLMENT CACHE
# *** CORE R7-A CHANGE ***
# ============================================================

def build_enrollment_cache(
    model_b0,
    model_b6,
    trials: List[dict],
) -> Tuple[
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    dict,
]:
    unique_enroll: List[Path] = []
    seen = set()

    for row in trials:
        p = row["enroll_path"]
        key = str(p)

        if key not in seen:
            seen.add(key)
            unique_enroll.append(p)

    print("\n" + "=" * 72)
    print("BUILDING ENROLLMENT CACHE — NOT TIMED")
    print("=" * 72)
    print(
        f"Unique enrollment utterances: {len(unique_enroll)}"
    )

    cache_b0: Dict[str, torch.Tensor] = {}
    cache_b6: Dict[str, torch.Tensor] = {}

    rss_before = current_rss_mb()

    t0 = time.perf_counter()

    for i, p in enumerate(unique_enroll, start=1):
        w, _ = audio(p, CFG.sample_rate)

        cache_b0[str(p)] = emb(
            model_b0,
            w,
        )

        cache_b6[str(p)] = emb(
            model_b6,
            w,
        )

        if (
            i == 1
            or i % 25 == 0
            or i == len(unique_enroll)
        ):
            print(
                f"  cached {i}/{len(unique_enroll)}"
            )

    cache_build_s = time.perf_counter() - t0

    rss_after = current_rss_mb()

    cache_bytes = 0

    for cache in (cache_b0, cache_b6):
        for tensor in cache.values():
            cache_bytes += (
                tensor.nelement()
                * tensor.element_size()
            )

    info = {
        "unique_enrollment_utterances": len(unique_enroll),
        "cache_build_time_s": float(cache_build_s),
        "embedding_tensor_bytes": int(cache_bytes),
        "embedding_tensor_mb": float(
            cache_bytes / (1024.0 ** 2)
        ),
        "rss_before_cache_mb": rss_before,
        "rss_after_cache_mb": rss_after,
        "rss_delta_cache_mb": rss_after - rss_before,
    }

    print(
        f"Cache build time: {cache_build_s:.2f} s"
    )
    print(
        f"Raw embedding tensor cache: "
        f"{info['embedding_tensor_mb']:.4f} MB"
    )
    print(
        f"RSS before/after: "
        f"{rss_before:.2f} / {rss_after:.2f} MB"
    )

    return cache_b0, cache_b6, info


# ============================================================
# TIMED PIPELINES
# Enrollment embeddings are already cached.
# ============================================================

def timed_static(
    model,
    enroll_embedding: torch.Tensor,
    test_path: Path,
) -> Tuple[float, float, float]:
    # IMPORTANT:
    # Timer starts BEFORE test audio loading.
    # Enrollment processing is NOT included.

    t0 = time.perf_counter()

    w_test, test_duration_s = audio(
        test_path,
        CFG.sample_rate,
    )

    z_test = emb(
        model,
        w_test,
    )

    s = cosine_score(
        enroll_embedding,
        z_test,
    )

    latency_ms = (
        time.perf_counter() - t0
    ) * 1000.0

    return s, latency_ms, test_duration_s


def timed_sequential_all(
    model_b0,
    model_b6,
    enroll_b0: torch.Tensor,
    enroll_b6: torch.Tensor,
    test_path: Path,
) -> Tuple[float, float, float, float]:
    # Preserve Stage10A sequential behavior:
    # b0 test audio is loaded/processed first,
    # then b6 test audio is loaded/processed again.
    #
    # Only enrollment side is removed from timing.

    t0 = time.perf_counter()

    w0, test_duration_s = audio(
        test_path,
        CFG.sample_rate,
    )

    z0 = emb(
        model_b0,
        w0,
    )

    s0 = cosine_score(
        enroll_b0,
        z0,
    )

    w6, _ = audio(
        test_path,
        CFG.sample_rate,
    )

    z6 = emb(
        model_b6,
        w6,
    )

    s6 = cosine_score(
        enroll_b6,
        z6,
    )

    latency_ms = (
        time.perf_counter() - t0
    ) * 1000.0

    # Sequential-all final verification score = b6 score.
    return (
        s0,
        s6,
        latency_ms,
        test_duration_s,
    )


def timed_frozen_b10(
    model_b0,
    model_b6,
    enroll_b0: torch.Tensor,
    enroll_b6: torch.Tensor,
    test_path: Path,
) -> Tuple[
    float,
    float,
    bool,
    float,
    float,
    float,
]:
    # Timer starts before b0 TEST audio processing.
    t0 = time.perf_counter()

    w0, test_duration_s = audio(
        test_path,
        CFG.sample_rate,
    )

    z0 = emb(
        model_b0,
        w0,
    )

    s0 = cosine_score(
        enroll_b0,
        z0,
    )

    route_t0 = time.perf_counter()

    margin = abs(
        s0 - CFG.tau_b0
    )

    routed = bool(
        margin <= CFG.delta_b10
    )

    route_overhead_ms = (
        time.perf_counter() - route_t0
    ) * 1000.0

    if routed:
        # Preserve canonical Stage10A behavior:
        # routed expert reloads/reprocesses TEST audio.
        w6, _ = audio(
            test_path,
            CFG.sample_rate,
        )

        z6 = emb(
            model_b6,
            w6,
        )

        s6 = cosine_score(
            enroll_b6,
            z6,
        )

        s_out = s6

    else:
        s6 = float("nan")
        s_out = s0

    latency_ms = (
        time.perf_counter() - t0
    ) * 1000.0

    return (
        s0,
        s_out,
        routed,
        latency_ms,
        test_duration_s,
        route_overhead_ms,
    )


# ============================================================
# PIPELINE BENCHMARK
# ============================================================

def benchmark_static(
    name: str,
    model,
    cache: Dict[str, torch.Tensor],
    trials: List[dict],
) -> Tuple[List[dict], dict]:
    print("\n" + "=" * 72)
    print(name)
    print("=" * 72)

    rows = []
    peak_rss = current_rss_mb()

    for i, trial in enumerate(trials, start=1):
        enroll_key = str(
            trial["enroll_path"]
        )

        s, latency_ms, test_duration_s = timed_static(
            model,
            cache[enroll_key],
            trial["test_path"],
        )

        rss = current_rss_mb()
        peak_rss = max(peak_rss, rss)

        rows.append(
            {
                "trial_index": trial["trial_index"],
                "label": trial["label"],
                "enroll_path": str(trial["enroll_path"]),
                "test_path": str(trial["test_path"]),
                "score": s,
                "latency_ms": latency_ms,
                "test_duration_s": test_duration_s,
                "test_rtf": (
                    latency_ms / 1000.0
                ) / test_duration_s,
            }
        )

        if (
            i == 1
            or i % 50 == 0
            or i == len(trials)
        ):
            print(
                f"  {i:4d}/{len(trials)} | "
                f"lat={latency_ms:.2f} ms"
            )

    labels = [
        r["label"]
        for r in rows
    ]

    scores = [
        r["score"]
        for r in rows
    ]

    latencies = [
        r["latency_ms"]
        for r in rows
    ]

    durations = [
        r["test_duration_s"]
        for r in rows
    ]

    result = {}
    result.update(
        verification_metrics(
            labels,
            scores,
        )
    )

    result.update(
        latency_summary(
            latencies,
            durations,
        )
    )

    result["max_rss_mb"] = float(
        peak_rss
    )

    return rows, result


def benchmark_seq_all(
    model_b0,
    model_b6,
    cache_b0: Dict[str, torch.Tensor],
    cache_b6: Dict[str, torch.Tensor],
    trials: List[dict],
) -> Tuple[List[dict], dict]:
    name = "SEQUENTIAL_ALL"

    print("\n" + "=" * 72)
    print(name)
    print("=" * 72)

    rows = []
    peak_rss = current_rss_mb()

    for i, trial in enumerate(trials, start=1):
        key = str(
            trial["enroll_path"]
        )

        (
            s0,
            s6,
            latency_ms,
            test_duration_s,
        ) = timed_sequential_all(
            model_b0,
            model_b6,
            cache_b0[key],
            cache_b6[key],
            trial["test_path"],
        )

        rss = current_rss_mb()
        peak_rss = max(peak_rss, rss)

        rows.append(
            {
                "trial_index": trial["trial_index"],
                "label": trial["label"],
                "enroll_path": str(trial["enroll_path"]),
                "test_path": str(trial["test_path"]),
                "score_b0": s0,
                "score_b6": s6,
                "score": s6,
                "latency_ms": latency_ms,
                "test_duration_s": test_duration_s,
                "test_rtf": (
                    latency_ms / 1000.0
                ) / test_duration_s,
            }
        )

        if (
            i == 1
            or i % 50 == 0
            or i == len(trials)
        ):
            print(
                f"  {i:4d}/{len(trials)} | "
                f"lat={latency_ms:.2f} ms"
            )

    labels = [
        r["label"]
        for r in rows
    ]

    scores = [
        r["score"]
        for r in rows
    ]

    latencies = [
        r["latency_ms"]
        for r in rows
    ]

    durations = [
        r["test_duration_s"]
        for r in rows
    ]

    result = {}

    result.update(
        verification_metrics(
            labels,
            scores,
        )
    )

    result.update(
        latency_summary(
            latencies,
            durations,
        )
    )

    result["max_rss_mb"] = float(
        peak_rss
    )

    return rows, result


def benchmark_b10(
    model_b0,
    model_b6,
    cache_b0: Dict[str, torch.Tensor],
    cache_b6: Dict[str, torch.Tensor],
    trials: List[dict],
) -> Tuple[List[dict], dict]:
    name = "FROZEN_B10"

    print("\n" + "=" * 72)
    print(name)
    print("=" * 72)

    rows = []
    peak_rss = current_rss_mb()

    for i, trial in enumerate(trials, start=1):
        key = str(
            trial["enroll_path"]
        )

        (
            s0,
            s_out,
            routed,
            latency_ms,
            test_duration_s,
            route_overhead_ms,
        ) = timed_frozen_b10(
            model_b0,
            model_b6,
            cache_b0[key],
            cache_b6[key],
            trial["test_path"],
        )

        rss = current_rss_mb()
        peak_rss = max(
            peak_rss,
            rss,
        )

        rows.append(
            {
                "trial_index": trial["trial_index"],
                "label": trial["label"],
                "enroll_path": str(trial["enroll_path"]),
                "test_path": str(trial["test_path"]),
                "score_b0": s0,
                "score": s_out,
                "routed_b6": int(routed),
                "latency_ms": latency_ms,
                "route_overhead_ms": route_overhead_ms,
                "test_duration_s": test_duration_s,
                "test_rtf": (
                    latency_ms / 1000.0
                ) / test_duration_s,
            }
        )

        if (
            i == 1
            or i % 50 == 0
            or i == len(trials)
        ):
            print(
                f"  {i:4d}/{len(trials)} | "
                f"route={int(routed)} | "
                f"lat={latency_ms:.2f} ms"
            )

    labels = [
        r["label"]
        for r in rows
    ]

    scores = [
        r["score"]
        for r in rows
    ]

    latencies = [
        r["latency_ms"]
        for r in rows
    ]

    durations = [
        r["test_duration_s"]
        for r in rows
    ]

    route_overheads = [
        r["route_overhead_ms"]
        for r in rows
    ]

    routed_count = int(
        sum(
            r["routed_b6"]
            for r in rows
        )
    )

    result = {}

    result.update(
        verification_metrics(
            labels,
            scores,
        )
    )

    result.update(
        latency_summary(
            latencies,
            durations,
        )
    )

    result.update(
        {
            "routed_count": routed_count,
            "b6_usage_percent": (
                100.0
                * routed_count
                / len(rows)
            ),
            "mean_route_overhead_ms": float(
                np.mean(route_overheads)
            ),
            "p95_route_overhead_ms": percentile(
                route_overheads,
                95,
            ),
            "max_rss_mb": float(
                peak_rss
            ),
        }
    )

    return rows, result


# ============================================================
# CSV OUTPUT
# ============================================================

def save_csv(
    path: Path,
    rows: List[dict],
):
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# INTEGRITY CHECKS
# ============================================================

def check_score_consistency(
    static_b0_rows: List[dict],
    static_b6_rows: List[dict],
    seq_rows: List[dict],
    b10_rows: List[dict],
) -> dict:
    # Same test-side inference should produce numerically identical
    # b0/b6 scores irrespective of pipeline.

    b0_static = np.asarray(
        [r["score"] for r in static_b0_rows],
        dtype=np.float64,
    )

    b6_static = np.asarray(
        [r["score"] for r in static_b6_rows],
        dtype=np.float64,
    )

    seq_b0 = np.asarray(
        [r["score_b0"] for r in seq_rows],
        dtype=np.float64,
    )

    seq_b6 = np.asarray(
        [r["score_b6"] for r in seq_rows],
        dtype=np.float64,
    )

    b10_b0 = np.asarray(
        [r["score_b0"] for r in b10_rows],
        dtype=np.float64,
    )

    max_abs_b0_static_vs_seq = float(
        np.max(
            np.abs(
                b0_static - seq_b0
            )
        )
    )

    max_abs_b0_static_vs_b10 = float(
        np.max(
            np.abs(
                b0_static - b10_b0
            )
        )
    )

    max_abs_b6_static_vs_seq = float(
        np.max(
            np.abs(
                b6_static - seq_b6
            )
        )
    )

    tolerance = 1e-6

    passed = bool(
        max_abs_b0_static_vs_seq <= tolerance
        and max_abs_b0_static_vs_b10 <= tolerance
        and max_abs_b6_static_vs_seq <= tolerance
    )

    return {
        "tolerance": tolerance,
        "max_abs_b0_static_vs_seq": max_abs_b0_static_vs_seq,
        "max_abs_b0_static_vs_b10": max_abs_b0_static_vs_b10,
        "max_abs_b6_static_vs_seq": max_abs_b6_static_vs_seq,
        "pass": passed,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    CFG.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.set_num_threads(
        CFG.torch_threads
    )

    try:
        torch.set_num_interop_threads(
            CFG.torch_interop_threads
        )
    except RuntimeError:
        # Can happen if interop threads were already initialized.
        pass

    print("=" * 72)
    print("R7-A — CACHED-ENROLLMENT RASPBERRY PI 5 BENCHMARK")
    print("=" * 72)

    print("\nProtocol:")
    print("  enrollment embedding extraction : NOT TIMED")
    print("  test audio I/O/preprocessing    : TIMED")
    print("  test embedding inference        : TIMED")
    print("  cosine scoring                  : TIMED")
    print("  Frozen-B10 routing              : TIMED")
    print("  routed b6 test inference        : TIMED")
    print("  cross-trial TEST cache          : DISABLED")

    print("\nFrozen policy:")
    print(
        f"  tau_b0    = {CFG.tau_b0:.9f}"
    )
    print(
        f"  delta_B10 = {CFG.delta_b10:.9f}"
    )

    print("\nEnvironment:")
    print(
        f"  Python             : {sys.version.split()[0]}"
    )
    print(
        f"  PyTorch            : {torch.__version__}"
    )
    print(
        f"  machine            : {platform.machine()}"
    )
    print(
        f"  platform           : {platform.platform()}"
    )
    print(
        f"  CPU count          : {os.cpu_count()}"
    )
    print(
        f"  torch threads      : {torch.get_num_threads()}"
    )
    print(
        f"  interop threads    : {torch.get_num_interop_threads()}"
    )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest = discover_manifest()

    manifest_sha = sha256_file(
        manifest
    )

    print("\nManifest:")
    print(
        f"  path   : {manifest}"
    )
    print(
        f"  SHA256 : {manifest_sha}"
    )

    if manifest_sha != CFG.expected_manifest_sha256:
        raise RuntimeError(
            "Manifest SHA256 gate FAILED."
        )

    trials = read_manifest(
        manifest
    )

    n_target = sum(
        int(r["label"] == 1)
        for r in trials
    )

    n_impostor = sum(
        int(r["label"] == 0)
        for r in trials
    )

    print(
        f"  trials : {len(trials)}"
    )
    print(
        f"  target : {n_target}"
    )
    print(
        f"  impostor: {n_impostor}"
    )

    if len(trials) != CFG.expected_trials:
        raise RuntimeError(
            "Trial-count gate FAILED."
        )

    if n_target != CFG.expected_target:
        raise RuntimeError(
            "Target-count gate FAILED."
        )

    if n_impostor != CFG.expected_impostor:
        raise RuntimeError(
            "Impostor-count gate FAILED."
        )

    print("  manifest integrity: PASS")

    # --------------------------------------------------------
    # Checkpoints
    # --------------------------------------------------------

    for name, p in (
        ("b0", CFG.b0_checkpoint),
        ("b6", CFG.b6_checkpoint),
    ):
        if not p.exists():
            raise FileNotFoundError(
                f"{name} checkpoint missing: {p}"
            )

        print(
            f"\n{name} checkpoint:"
        )
        print(
            f"  {p}"
        )
        print(
            f"  SHA256: {sha256_file(p)}"
        )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    print("\nLoading b0...")
    model_b0 = load_model(
        CFG.b0_checkpoint
    )

    print("Loading b6...")
    model_b6 = load_model(
        CFG.b6_checkpoint
    )

    b0_params = sum(
        p.numel()
        for p in model_b0.parameters()
    )

    b6_params = sum(
        p.numel()
        for p in model_b6.parameters()
    )

    print(
        f"b0 params: {b0_params:,}"
    )
    print(
        f"b6 params: {b6_params:,}"
    )

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    warmup_models(
        model_b0,
        model_b6,
        trials,
    )

    # --------------------------------------------------------
    # Enrollment cache — OUTSIDE benchmark timing
    # --------------------------------------------------------

    (
        cache_b0,
        cache_b6,
        cache_info,
    ) = build_enrollment_cache(
        model_b0,
        model_b6,
        trials,
    )

    # --------------------------------------------------------
    # Benchmarks
    # --------------------------------------------------------

    static_b0_rows, static_b0 = benchmark_static(
        "STATIC_B0",
        model_b0,
        cache_b0,
        trials,
    )

    static_b6_rows, static_b6 = benchmark_static(
        "STATIC_B6",
        model_b6,
        cache_b6,
        trials,
    )

    seq_rows, seq_all = benchmark_seq_all(
        model_b0,
        model_b6,
        cache_b0,
        cache_b6,
        trials,
    )

    b10_rows, frozen_b10 = benchmark_b10(
        model_b0,
        model_b6,
        cache_b0,
        cache_b6,
        trials,
    )

    # --------------------------------------------------------
    # Integrity
    # --------------------------------------------------------

    score_integrity = check_score_consistency(
        static_b0_rows,
        static_b6_rows,
        seq_rows,
        b10_rows,
    )

    # --------------------------------------------------------
    # Direct measured savings
    # --------------------------------------------------------

    b10_mean = frozen_b10[
        "mean_latency_ms"
    ]

    seq_mean = seq_all[
        "mean_latency_ms"
    ]

    b6_mean = static_b6[
        "mean_latency_ms"
    ]

    saving_vs_seq = (
        100.0
        * (
            1.0
            - b10_mean / seq_mean
        )
    )

    saving_vs_b6 = (
        100.0
        * (
            1.0
            - b10_mean / b6_mean
        )
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    save_csv(
        CFG.output_dir / "r7a_static_b0_trials.csv",
        static_b0_rows,
    )

    save_csv(
        CFG.output_dir / "r7a_static_b6_trials.csv",
        static_b6_rows,
    )

    save_csv(
        CFG.output_dir / "r7a_sequential_all_trials.csv",
        seq_rows,
    )

    save_csv(
        CFG.output_dir / "r7a_frozen_b10_trials.csv",
        b10_rows,
    )

    report = {
        "experiment": (
            "R7-A — Cached-Enrollment Raspberry Pi 5 Benchmark"
        ),

        "protocol": {
            "enrollment_embeddings": (
                "precomputed_and_cached_before_timing"
            ),
            "cached_models": [
                "ReDimNet-b0",
                "ReDimNet-b6",
            ],
            "test_audio_io": "timed",
            "test_preprocessing": "timed",
            "test_embedding_inference": "timed",
            "cosine_scoring": "timed",
            "routing": "timed",
            "expert_test_inference_if_routed": "timed",
            "cross_trial_test_embedding_cache": "DISABLED",
            "rtf_definition": (
                "latency_seconds / test_duration_seconds"
            ),
        },

        "config": {
            **asdict(CFG),
            "redimnet_source": str(CFG.redimnet_source),
            "b0_checkpoint": str(CFG.b0_checkpoint),
            "b6_checkpoint": str(CFG.b6_checkpoint),
            "project_root": str(CFG.project_root),
            "output_dir": str(CFG.output_dir),
        },

        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "machine": platform.machine(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
        },

        "manifest": {
            "path": str(manifest),
            "sha256": manifest_sha,
            "trials": len(trials),
            "target": n_target,
            "impostor": n_impostor,
        },

        "checkpoints": {
            "b0_sha256": sha256_file(
                CFG.b0_checkpoint
            ),
            "b6_sha256": sha256_file(
                CFG.b6_checkpoint
            ),
            "b0_params": b0_params,
            "b6_params": b6_params,
        },

        "enrollment_cache": cache_info,

        "results": {
            "STATIC_B0": static_b0,
            "STATIC_B6": static_b6,
            "SEQUENTIAL_ALL": seq_all,
            "FROZEN_B10": frozen_b10,
        },

        "direct_measured_savings": {
            "b10_vs_sequential_all_percent": float(
                saving_vs_seq
            ),
            "b10_vs_static_b6_percent": float(
                saving_vs_b6
            ),
        },

        "score_integrity": score_integrity,

        "historical_reference_only_not_gate": {
            "stage10a_b10_usage_percent": 8.0,
            "stage10a_b10_eer_percent": 0.8,
            "stage10a_uncached_b10_mean_latency_ms": 3369.37,
            "stage10a_uncached_seq_all_mean_latency_ms": 20132.68,
            "stage10a_uncached_static_b6_mean_latency_ms": 18235.99,
            "note": (
                "Historical Stage10A values are reference only. "
                "Latency must not be compared as if timing boundaries "
                "were identical."
            ),
        },
    }

    report_path = (
        CFG.output_dir
        / "r7a_cached_enrollment_report.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    # --------------------------------------------------------
    # FINAL CONSOLE SUMMARY
    # --------------------------------------------------------

    print("\n")
    print("=" * 72)
    print("R7-A FINAL RESULTS")
    print("=" * 72)

    def show(name: str, r: dict):
        print(f"\n{name}")
        print(
            f"  EER       : {r['eer_percent']:.6f}%"
        )
        print(
            f"  minDCF    : {r['min_dcf']:.9f}"
        )
        print(
            f"  AUC       : {r['auc']:.9f}"
        )
        print(
            f"  mean      : {r['mean_latency_ms']:.2f} ms"
        )
        print(
            f"  median    : {r['median_latency_ms']:.2f} ms"
        )
        print(
            f"  P95       : {r['p95_latency_ms']:.2f} ms"
        )
        print(
            f"  P99       : {r['p99_latency_ms']:.2f} ms"
        )
        print(
            f"  mean RTF  : {r['mean_test_rtf']:.4f}"
        )
        print(
            f"  max RSS   : {r['max_rss_mb']:.2f} MB"
        )

    show(
        "STATIC_B0",
        static_b0,
    )

    show(
        "STATIC_B6",
        static_b6,
    )

    show(
        "SEQUENTIAL_ALL",
        seq_all,
    )

    show(
        "FROZEN_B10",
        frozen_b10,
    )

    print("\nFrozen-B10 routing")
    print(
        f"  routed    : "
        f"{frozen_b10['routed_count']}/{len(trials)}"
    )
    print(
        f"  b6 usage  : "
        f"{frozen_b10['b6_usage_percent']:.6f}%"
    )
    print(
        f"  route mean: "
        f"{frozen_b10['mean_route_overhead_ms']:.6f} ms"
    )

    print("\nDIRECT MEASURED SAVINGS")
    print(
        f"  B10 vs Sequential-All : "
        f"{saving_vs_seq:.4f}%"
    )
    print(
        f"  B10 vs Static-b6      : "
        f"{saving_vs_b6:.4f}%"
    )

    print("\nSCORE INTEGRITY")
    print(
        f"  b0 static vs seq max |Δ| : "
        f"{score_integrity['max_abs_b0_static_vs_seq']:.3e}"
    )
    print(
        f"  b0 static vs B10 max |Δ| : "
        f"{score_integrity['max_abs_b0_static_vs_b10']:.3e}"
    )
    print(
        f"  b6 static vs seq max |Δ| : "
        f"{score_integrity['max_abs_b6_static_vs_seq']:.3e}"
    )
    print(
        "  STATUS: "
        + (
            "PASS"
            if score_integrity["pass"]
            else "FAIL"
        )
    )

    print("\nENROLLMENT CACHE")
    print(
        f"  unique enrollment utts : "
        f"{cache_info['unique_enrollment_utterances']}"
    )
    print(
        f"  raw embedding cache    : "
        f"{cache_info['embedding_tensor_mb']:.4f} MB"
    )
    print(
        f"  cache build time       : "
        f"{cache_info['cache_build_time_s']:.2f} s "
        f"(NOT TIMED)"
    )

    print("\nArtifacts:")
    print(
        f"  {CFG.output_dir}"
    )
    print(
        f"  {report_path}"
    )

    print("\nR7-A execution complete.")
    print(
        "Do NOT LOCK scientifically until output has been audited."
    )


if __name__ == "__main__":
    main()