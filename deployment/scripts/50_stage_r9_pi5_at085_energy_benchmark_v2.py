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
        "r9_at085_energy_benchmark"
    )

    # Persistent enrollment cache for R9 repeat runs.
    # Cache BUILD/LOAD remains outside the AT085 measurement boundary.
    disk_cache_path: Path = Path(
        "./"
        "results/reviewer_audit/"
        "r9_at085_energy_benchmark/"
        "r9_enrollment_cache_b0_b6.pt"
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
        f"Audio file not found under canonical VoxCeleb1-O root:\n{q}"
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
                    f"Enrollment audio missing:\n{enroll_path}"
                )

            if not test_path.exists():
                raise FileNotFoundError(
                    f"Test audio missing:\n{test_path}"
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
# R9-V2 PERSISTENT ENROLLMENT CACHE
# Cache build/load is OUTSIDE the AT085 measurement boundary.
# ============================================================

def unique_enrollment_keys(trials: List[dict]) -> List[str]:
    keys = []
    seen = set()

    for row in trials:
        key = str(row["enroll_path"])

        if key not in seen:
            seen.add(key)
            keys.append(key)

    return keys


def save_persistent_enrollment_cache(
    path: Path,
    cache_b0: Dict[str, torch.Tensor],
    cache_b6: Dict[str, torch.Tensor],
    manifest_sha256: str,
    b0_sha256: str,
    b6_sha256: str,
    trials: List[dict],
):
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "format_version": 1,
        "manifest_sha256": manifest_sha256,
        "b0_checkpoint_sha256": b0_sha256,
        "b6_checkpoint_sha256": b6_sha256,
        "sample_rate": CFG.sample_rate,
        "unique_enrollment_keys": unique_enrollment_keys(trials),
        "cache_b0": {
            k: v.detach().cpu()
            for k, v in cache_b0.items()
        },
        "cache_b6": {
            k: v.detach().cpu()
            for k, v in cache_b6.items()
        },
    }

    torch.save(payload, path)

    print("\nPersistent enrollment cache SAVED:")
    print(f"  {path}")
    print(f"  file size: {path.stat().st_size / (1024.0 ** 2):.4f} MB")


def load_persistent_enrollment_cache(
    path: Path,
    manifest_sha256: str,
    b0_sha256: str,
    b6_sha256: str,
    trials: List[dict],
):
    if not path.exists():
        return None

    print("\n" + "=" * 72)
    print("CHECKING PERSISTENT ENROLLMENT CACHE — NOT TIMED")
    print("=" * 72)
    print(f"Path: {path}")

    t0 = time.perf_counter()

    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as exc:
        print(f"Persistent cache load FAILED: {exc}")
        print("Cache will be rebuilt from audio.")
        return None

    expected_keys = unique_enrollment_keys(trials)

    checks = {
        "format_version": payload.get("format_version") == 1,
        "manifest_sha256": (
            payload.get("manifest_sha256") == manifest_sha256
        ),
        "b0_checkpoint_sha256": (
            payload.get("b0_checkpoint_sha256") == b0_sha256
        ),
        "b6_checkpoint_sha256": (
            payload.get("b6_checkpoint_sha256") == b6_sha256
        ),
        "sample_rate": (
            int(payload.get("sample_rate", -1)) == CFG.sample_rate
        ),
        "enrollment_keys": (
            payload.get("unique_enrollment_keys") == expected_keys
        ),
    }

    for name, ok in checks.items():
        print(f"  {name:24s}: {'PASS' if ok else 'FAIL'}")

    if not all(checks.values()):
        print("Persistent cache integrity FAILED.")
        print("Cache will be rebuilt from audio.")
        return None

    cache_b0 = payload.get("cache_b0")
    cache_b6 = payload.get("cache_b6")

    if not isinstance(cache_b0, dict) or not isinstance(cache_b6, dict):
        print("Persistent cache tensor dictionaries are invalid.")
        print("Cache will be rebuilt from audio.")
        return None

    if set(cache_b0.keys()) != set(expected_keys):
        print("b0 persistent-cache key set FAILED.")
        return None

    if set(cache_b6.keys()) != set(expected_keys):
        print("b6 persistent-cache key set FAILED.")
        return None

    cache_bytes = 0

    for cache in (cache_b0, cache_b6):
        for tensor in cache.values():
            if not isinstance(tensor, torch.Tensor):
                print("Persistent cache contains a non-tensor value.")
                return None

            cache_bytes += (
                tensor.nelement()
                * tensor.element_size()
            )

    load_s = time.perf_counter() - t0

    info = {
        "unique_enrollment_utterances": len(expected_keys),
        "cache_source": "persistent_disk_cache",
        "cache_build_time_s": 0.0,
        "cache_load_time_s": float(load_s),
        "embedding_tensor_bytes": int(cache_bytes),
        "embedding_tensor_mb": float(
            cache_bytes / (1024.0 ** 2)
        ),
        "persistent_cache_path": str(path),
        "persistent_cache_file_mb": float(
            path.stat().st_size / (1024.0 ** 2)
        ),
    }

    print("Persistent enrollment cache integrity: PASS")
    print(f"Cache load time: {load_s:.3f} s")
    print(
        f"Raw embedding tensors: "
        f"{info['embedding_tensor_mb']:.4f} MB"
    )

    return cache_b0, cache_b6, info


def get_or_build_enrollment_cache(
    model_b0,
    model_b6,
    trials: List[dict],
    manifest_sha256: str,
    b0_sha256: str,
    b6_sha256: str,
):
    loaded = load_persistent_enrollment_cache(
        CFG.disk_cache_path,
        manifest_sha256,
        b0_sha256,
        b6_sha256,
        trials,
    )

    if loaded is not None:
        return loaded

    cache_b0, cache_b6, info = build_enrollment_cache(
        model_b0,
        model_b6,
        trials,
    )

    info["cache_source"] = "rebuilt_from_audio"
    info["cache_load_time_s"] = 0.0
    info["persistent_cache_path"] = str(CFG.disk_cache_path)

    save_persistent_enrollment_cache(
        CFG.disk_cache_path,
        cache_b0,
        cache_b6,
        manifest_sha256,
        b0_sha256,
        b6_sha256,
        trials,
    )

    info["persistent_cache_file_mb"] = float(
        CFG.disk_cache_path.stat().st_size / (1024.0 ** 2)
    )

    return cache_b0, cache_b6, info


# ============================================================
# R9-V2 TEMPERATURE LOGGING
# ============================================================

def read_pi_temperature_c() -> float | None:
    candidates = (
        Path("/sys/class/thermal/thermal_zone0/temp"),
        Path("/sys/devices/virtual/thermal/thermal_zone0/temp"),
    )

    for path in candidates:
        try:
            raw = path.read_text(encoding="utf-8").strip()
            value = float(raw)

            # Linux thermal sysfs normally reports millidegrees C.
            if value > 1000.0:
                value /= 1000.0

            if -20.0 <= value <= 150.0:
                return float(value)
        except (OSError, ValueError):
            pass

    return None


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
# R9 INTERACTIVE AT085 ENERGY BENCHMARK
# Derived from validated R7-A cached-enrollment implementation.
# ============================================================


# ============================================================
# R9 — AT085 ENERGY HELPERS
# ============================================================

PIPELINES = {
    "1": "STATIC_B0",
    "2": "STATIC_B6",
    "3": "SEQUENTIAL_ALL",
    "4": "FROZEN_B10",
}


def select_pipeline() -> str | None:
    print("\n" + "=" * 72)
    print("R9 — SELECT PIPELINE")
    print("=" * 72)
    print("  1 — Static-b0")
    print("  2 — Static-b6")
    print("  3 — Sequential-All")
    print("  4 — Frozen-B10")
    print("  0 — Finish")

    while True:
        choice = input("\nChoice: ").strip()

        if choice == "0":
            return None

        if choice in PIPELINES:
            return PIPELINES[choice]

        print("Invalid choice. Enter 0, 1, 2, 3, or 4.")


def read_wh(prompt: str) -> float:
    while True:
        raw = input(prompt).strip().replace(",", ".")

        try:
            value = float(raw)
        except ValueError:
            print("Please enter only the AT085 Wh value, e.g. 2.347")
            continue

        if value < 0:
            print("Wh cannot be negative.")
            continue

        return value


def next_run_number(csv_path: Path, pipeline: str) -> int:
    if not csv_path.exists():
        return 1

    run_numbers = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row.get("pipeline") == pipeline:
                try:
                    run_numbers.append(int(row["run_number"]))
                except (KeyError, TypeError, ValueError):
                    pass

    return max(run_numbers, default=0) + 1


def append_energy_summary(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()

    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(row.keys()),
        )

        if not exists:
            writer.writeheader()

        writer.writerow(row)


def run_selected_pipeline(
    pipeline: str,
    model_b0,
    model_b6,
    cache_b0: Dict[str, torch.Tensor],
    cache_b6: Dict[str, torch.Tensor],
    trials: List[dict],
):
    if pipeline == "STATIC_B0":
        return benchmark_static(
            "STATIC_B0",
            model_b0,
            cache_b0,
            trials,
        )

    if pipeline == "STATIC_B6":
        return benchmark_static(
            "STATIC_B6",
            model_b6,
            cache_b6,
            trials,
        )

    if pipeline == "SEQUENTIAL_ALL":
        return benchmark_seq_all(
            model_b0,
            model_b6,
            cache_b0,
            cache_b6,
            trials,
        )

    if pipeline == "FROZEN_B10":
        return benchmark_b10(
            model_b0,
            model_b6,
            cache_b0,
            cache_b6,
            trials,
        )

    raise ValueError(f"Unknown pipeline: {pipeline}")


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
        pass

    print("=" * 72)
    print("R9-V2 — RASPBERRY PI 5 / AT085 SYSTEM-LEVEL ENERGY BENCHMARK")
    print("=" * 72)

    print("\nMeasurement boundary:")
    print("  model loading                    : EXCLUDED")
    print("  warm-up                          : EXCLUDED")
    print("  enrollment embedding cache      : EXCLUDED")
    print("  test audio I/O/preprocessing    : INCLUDED")
    print("  test embedding inference        : INCLUDED")
    print("  cosine scoring                  : INCLUDED")
    print("  Frozen-B10 routing              : INCLUDED")
    print("  routed b6 test inference        : INCLUDED")
    print("  cross-trial TEST cache          : DISABLED")
    print("  AT085 measurement               : Pi 5 power input")

    print("\nFrozen policy:")
    print(f"  tau_b0    = {CFG.tau_b0:.9f}")
    print(f"  delta_B10 = {CFG.delta_b10:.9f}")

    print("\nEnvironment:")
    print(f"  Python          : {sys.version.split()[0]}")
    print(f"  PyTorch         : {torch.__version__}")
    print(f"  machine         : {platform.machine()}")
    print(f"  platform        : {platform.platform()}")
    print(f"  CPU count       : {os.cpu_count()}")
    print(f"  torch threads   : {torch.get_num_threads()}")
    print(f"  interop threads : {torch.get_num_interop_threads()}")

    # --------------------------------------------------------
    # Canonical manifest
    # --------------------------------------------------------

    manifest = discover_manifest()
    manifest_sha = sha256_file(manifest)

    print("\nManifest:")
    print(f"  path   : {manifest}")
    print(f"  SHA256 : {manifest_sha}")

    if manifest_sha != CFG.expected_manifest_sha256:
        raise RuntimeError("Manifest SHA256 gate FAILED.")

    trials = read_manifest(manifest)

    n_target = sum(int(r["label"] == 1) for r in trials)
    n_impostor = sum(int(r["label"] == 0) for r in trials)

    print(f"  trials   : {len(trials)}")
    print(f"  target   : {n_target}")
    print(f"  impostor : {n_impostor}")

    if len(trials) != CFG.expected_trials:
        raise RuntimeError("Trial-count gate FAILED.")

    if n_target != CFG.expected_target:
        raise RuntimeError("Target-count gate FAILED.")

    if n_impostor != CFG.expected_impostor:
        raise RuntimeError("Impostor-count gate FAILED.")

    print("  manifest integrity: PASS")

    # --------------------------------------------------------
    # Checkpoints and models
    # --------------------------------------------------------

    for name, p in (
        ("b0", CFG.b0_checkpoint),
        ("b6", CFG.b6_checkpoint),
    ):
        if not p.exists():
            raise FileNotFoundError(
                f"{name} checkpoint missing: {p}"
            )

    b0_sha = sha256_file(CFG.b0_checkpoint)
    b6_sha = sha256_file(CFG.b6_checkpoint)

    print("\nb0 checkpoint:")
    print(f"  {CFG.b0_checkpoint}")
    print(f"  SHA256: {b0_sha}")

    print("\nb6 checkpoint:")
    print(f"  {CFG.b6_checkpoint}")
    print(f"  SHA256: {b6_sha}")

    print("\nLoading b0...")
    model_b0 = load_model(CFG.b0_checkpoint)

    print("Loading b6...")
    model_b6 = load_model(CFG.b6_checkpoint)

    b0_params = sum(p.numel() for p in model_b0.parameters())
    b6_params = sum(p.numel() for p in model_b6.parameters())

    print(f"b0 params: {b0_params:,}")
    print(f"b6 params: {b6_params:,}")

    # --------------------------------------------------------
    # Warm-up and enrollment cache are OUTSIDE energy boundary
    # --------------------------------------------------------

    warmup_models(
        model_b0,
        model_b6,
        trials,
    )

    (
        cache_b0,
        cache_b6,
        cache_info,
    ) = get_or_build_enrollment_cache(
        model_b0,
        model_b6,
        trials,
        manifest_sha,
        b0_sha,
        b6_sha,
    )

    print("\n" + "=" * 72)
    print("R9 PREPARATION COMPLETE")
    print("=" * 72)
    print(
        f"Enrollment cache ready: "
        f"{cache_info['unique_enrollment_utterances']} unique utterances"
    )
    print(
        f"Cache source: "
        f"{cache_info.get('cache_source', 'unknown')}"
    )

    if cache_info.get("cache_source") == "persistent_disk_cache":
        print(
            f"Cache load time: "
            f"{cache_info.get('cache_load_time_s', 0.0):.3f} s — EXCLUDED"
        )
    else:
        print(
            f"Cache build time: "
            f"{cache_info.get('cache_build_time_s', 0.0):.2f} s — EXCLUDED"
        )

    print("\nThe AT085 measured run starts only after START Wh is entered.")

    summary_csv = CFG.output_dir / "r9_at085_energy_summary.csv"

    # --------------------------------------------------------
    # Interactive pipeline loop
    # --------------------------------------------------------

    while True:
        pipeline = select_pipeline()

        if pipeline is None:
            print("\nR9 session finished.")
            print(f"Summary CSV: {summary_csv}")
            break

        run_number = next_run_number(
            summary_csv,
            pipeline,
        )

        print("\n" + "=" * 72)
        print(f"R9 ENERGY RUN — {pipeline} — RUN {run_number}")
        print("=" * 72)
        print(f"Trials   : {len(trials)}")
        print(f"Target   : {n_target}")
        print(f"Impostor : {n_impostor}")
        print("Enrollment cache: READY / EXCLUDED FROM MEASUREMENT")

        print("\nAT085:")
        print("  1. Read the accumulated ENERGY value in Wh.")
        print("  2. Enter that value below.")
        print("  3. After entering it, the benchmark starts immediately.")
        print("  4. Do not run other heavy tasks during the benchmark.")

        temp_start_c = read_pi_temperature_c()

        if temp_start_c is None:
            print("\nPi temperature before run: unavailable")
        else:
            print(
                f"\nPi temperature before run: "
                f"{temp_start_c:.2f} C"
            )

        start_wh = read_wh("\nAT085 START Wh = ")

        print("\nSTART accepted.")
        print("Benchmark starts NOW.\n")

        wall_t0 = time.perf_counter()

        trial_rows, result = run_selected_pipeline(
            pipeline,
            model_b0,
            model_b6,
            cache_b0,
            cache_b6,
            trials,
        )

        wall_seconds = time.perf_counter() - wall_t0
        temp_end_c = read_pi_temperature_c()

        print("\n" + "=" * 72)
        print(f"{pipeline} COMPLETE")
        print("=" * 72)
        print(f"500-trial wall time : {wall_seconds:.3f} s")

        if temp_end_c is None:
            print("Pi temperature end  : unavailable")
        else:
            print(f"Pi temperature end  : {temp_end_c:.2f} C")

        print("\nRead the AT085 accumulated Wh value NOW.")

        end_wh = read_wh("AT085 END Wh   = ")

        energy_wh = end_wh - start_wh

        if energy_wh <= 0:
            print(
                "\nWARNING: END Wh must be greater than START Wh. "
                "This run will NOT be saved."
            )
            continue

        energy_j = energy_wh * 3600.0
        energy_j_per_trial = energy_j / len(trials)
        average_power_w = energy_j / wall_seconds

        routed_count = ""
        b6_usage_percent = ""

        if pipeline == "FROZEN_B10":
            routed_count = int(result["routed_count"])
            b6_usage_percent = float(
                result["b6_usage_percent"]
            )

        timestamp = time.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        row = {
            "timestamp": timestamp,
            "pipeline": pipeline,
            "run_number": run_number,
            "trials": len(trials),
            "target": n_target,
            "impostor": n_impostor,
            "start_wh": start_wh,
            "end_wh": end_wh,
            "energy_wh": energy_wh,
            "energy_j": energy_j,
            "energy_j_per_trial": energy_j_per_trial,
            "wall_seconds": wall_seconds,
            "average_system_power_w": average_power_w,
            "benchmark_mean_latency_ms": result["mean_latency_ms"],
            "benchmark_median_latency_ms": result["median_latency_ms"],
            "benchmark_p95_latency_ms": result["p95_latency_ms"],
            "benchmark_p99_latency_ms": result["p99_latency_ms"],
            "eer_percent": result["eer_percent"],
            "min_dcf": result["min_dcf"],
            "auc": result["auc"],
            "routed_count": routed_count,
            "b6_usage_percent": b6_usage_percent,
            "pi_temp_start_c": temp_start_c,
            "pi_temp_end_c": temp_end_c,
            "pi_temp_delta_c": (
                None
                if temp_start_c is None or temp_end_c is None
                else temp_end_c - temp_start_c
            ),
            "cache_source": cache_info.get("cache_source", "unknown"),
            "manifest_sha256": manifest_sha,
            "tau_b0": CFG.tau_b0,
            "delta_b10": CFG.delta_b10,
        }

        append_energy_summary(
            summary_csv,
            row,
        )

        run_json = (
            CFG.output_dir
            / f"{pipeline.lower()}_run{run_number}.json"
        )

        payload = {
            "experiment": (
                "R9 — Raspberry Pi 5 AT085 "
                "system-level energy benchmark"
            ),
            "measurement_scope": (
                "system-level energy measured at "
                "the Raspberry Pi 5 power input"
            ),
            "pipeline": pipeline,
            "run_number": run_number,
            "energy": row,
            "verification_and_latency": result,
            "manifest": {
                "path": str(manifest),
                "sha256": manifest_sha,
            },
            "checkpoints": {
                "b0_path": str(CFG.b0_checkpoint),
                "b0_sha256": b0_sha,
                "b6_path": str(CFG.b6_checkpoint),
                "b6_sha256": b6_sha,
                "b0_params": b0_params,
                "b6_params": b6_params,
            },
            "enrollment_cache": cache_info,
            "protocol": {
                "model_loading_energy": "excluded",
                "warmup_energy": "excluded",
                "enrollment_cache_energy": "excluded",
                "test_audio_io_preprocessing_energy": "included",
                "test_inference_energy": "included",
                "cosine_scoring_energy": "included",
                "routing_energy": "included_when_applicable",
                "routed_b6_inference_energy": "included_when_applicable",
                "cross_trial_test_cache": "disabled",
                "at085_reset_required": False,
            },
        }

        with run_json.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                payload,
                f,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

        # Per-trial output retained for auditability.
        trial_csv = (
            CFG.output_dir
            / f"{pipeline.lower()}_run{run_number}_trials.csv"
        )

        save_csv(
            trial_csv,
            trial_rows,
        )

        print("\n" + "=" * 72)
        print("R9 ENERGY RESULT")
        print("=" * 72)
        print(f"Pipeline            : {pipeline}")
        print(f"Run                 : {run_number}")
        print(f"AT085 START         : {start_wh:.6f} Wh")
        print(f"AT085 END           : {end_wh:.6f} Wh")
        print(f"Energy              : {energy_wh:.6f} Wh")
        print(f"Energy              : {energy_j:.3f} J")
        print(f"Energy / trial      : {energy_j_per_trial:.6f} J/trial")
        print(f"Wall time           : {wall_seconds:.3f} s")
        print(f"Average system power: {average_power_w:.6f} W")

        if temp_start_c is not None:
            print(f"Pi temp START       : {temp_start_c:.2f} C")

        if temp_end_c is not None:
            print(f"Pi temp END         : {temp_end_c:.2f} C")

        if temp_start_c is not None and temp_end_c is not None:
            print(
                f"Pi temp delta       : "
                f"{temp_end_c - temp_start_c:+.2f} C"
            )

        if pipeline == "FROZEN_B10":
            print(
                f"b6 routed           : "
                f"{routed_count}/{len(trials)}"
            )
            print(
                f"b6 usage            : "
                f"{b6_usage_percent:.6f}%"
            )

            if routed_count != 40:
                print(
                    "WARNING: Frozen-B10 routed count differs "
                    "from the canonical R7-A 40/500 result."
                )
            else:
                print(
                    "Frozen-B10 routing integrity: PASS (40/500)"
                )

        print("\nSaved:")
        print(f"  {summary_csv}")
        print(f"  {run_json}")
        print(f"  {trial_csv}")

        print(
            "\nYou may now select another pipeline. "
            "Wait for the Pi thermal state to stabilize before the next run."
        )


if __name__ == "__main__":
    main()
