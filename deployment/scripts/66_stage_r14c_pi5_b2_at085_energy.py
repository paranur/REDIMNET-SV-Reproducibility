# 66_stage_r14c_pi5_b2_at085_energy.py
#
# R14-C — Raspberry Pi 5 Static-b2 AT085 Energy Benchmark
#
# Exact extension of the validated R9-V2 cached-enrollment
# system-level energy protocol to ReDimNet-b2.
#
# Measurement boundary:
#   model loading                 EXCLUDED
#   warm-up                       EXCLUDED
#   enrollment cache             EXCLUDED
#   test audio I/O/preprocessing INCLUDED
#   test embedding inference     INCLUDED
#   cosine scoring               INCLUDED
#
# Cross-trial test embedding cache is DISABLED.

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly
from sklearn.metrics import roc_auc_score, roc_curve


@dataclass(frozen=True)
class Config:
    redimnet_source: Path = Path(
        "external/ReDimNet"
    )

    b2_checkpoint: Path = Path(
        "models/redimnet/b2-vox2-ptn.pt"
    )

    project_root: Path = Path(
        "."
    )

    output_dir: Path = Path(
        "./"
        "results/reviewer_audit/"
        "r14c_pi5_b2_at085_energy"
    )

    disk_cache_path: Path = Path(
        "./"
        "results/reviewer_audit/"
        "r14c_pi5_b2_at085_energy/"
        "r14c_enrollment_cache_b2.pt"
    )

    sample_rate: int = 16000

    expected_trials: int = 500
    expected_target: int = 250
    expected_impostor: int = 250

    expected_manifest_sha256: str = (
        "d4e26be792f1ebcb2bf83719d1ee303399cde0dbb299248b6258b4ad436aa2c0"
    )

    expected_b2_sha256: str = (
        "c18a42926878bc8ac079623fbf36f0bc8054cda1199e96fbe1a3f8e131796647"
    )

    torch_threads: int = 4
    torch_interop_threads: int = 1

    warmup_files: int = 5


CFG = Config()


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def audio(
    path: Path,
    target_sr: int,
) -> Tuple[torch.Tensor, float]:

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

        x = resample_poly(
            x,
            int(target_sr) // g,
            int(sr) // g,
        ).astype(
            np.float32,
            copy=False,
        )

    duration_s = float(
        len(x) / target_sr
    )

    w = torch.from_numpy(
        np.ascontiguousarray(x)
    ).float().unsqueeze(0)

    return w, duration_s


def load_model(checkpoint: Path):
    if str(CFG.redimnet_source) not in sys.path:
        sys.path.insert(
            0,
            str(CFG.redimnet_source),
        )

    from redimnet.model import ReDimNetWrap

    full = torch.load(
        str(checkpoint),
        map_location="cpu",
        weights_only=False,
    )

    if "model_config" not in full:
        raise RuntimeError(
            "model_config missing."
        )

    if "state_dict" not in full:
        raise RuntimeError(
            "state_dict missing."
        )

    model = ReDimNetWrap(
        **full["model_config"]
    )

    model.load_state_dict(
        full["state_dict"],
        strict=True,
    )

    model.eval()
    model.cpu()

    return model


@torch.inference_mode()
def emb(
    model,
    w: torch.Tensor,
) -> torch.Tensor:

    z = model(w)

    if isinstance(z, (tuple, list)):
        z = z[0]

    if isinstance(z, dict):
        for key in (
            "embedding",
            "embeddings",
            "x",
            "output",
        ):
            if key in z:
                z = z[key]
                break
        else:
            raise RuntimeError(
                f"Unsupported model output: "
                f"{list(z.keys())}"
            )

    if z.ndim == 3:
        z = z.mean(dim=1)

    if z.ndim == 1:
        z = z.unsqueeze(0)

    return F.normalize(
        z,
        p=2,
        dim=1,
    ).cpu()


def cosine_score(
    a: torch.Tensor,
    b: torch.Tensor,
) -> float:

    return float(
        F.cosine_similarity(
            a,
            b,
            dim=1,
        ).item()
    )


# ============================================================
# MANIFEST
# ============================================================

def discover_manifest() -> Path:
    results_root = (
        CFG.project_root / "results"
    )

    for p in results_root.rglob("*.csv"):
        try:
            if (
                sha256_file(p)
                == CFG.expected_manifest_sha256
            ):
                return p
        except OSError:
            pass

    raise FileNotFoundError(
        "Canonical 500-trial manifest "
        "not found."
    )


def normalize_label(value: str) -> int:
    s = str(value).strip().lower()

    if s in {
        "1", "target", "same",
        "positive", "pos", "true",
    }:
        return 1

    if s in {
        "0", "-1", "nontarget",
        "non-target", "different",
        "impostor", "negative",
        "neg", "false",
    }:
        return 0

    raise ValueError(
        f"Unknown label: {value}"
    )


def first_existing(
    row: Dict[str, str],
    keys,
):
    for key in keys:
        if (
            key in row
            and row[key] is not None
            and str(row[key]).strip()
        ):
            return str(row[key]).strip()

    return None


def resolve_audio_path(raw: str) -> Path:
    p = Path(raw)

    if p.is_absolute():
        return p

    q = (
        CFG.project_root
        / "DATASETS"
        / "VoxCeleb1_O"
        / p
    )

    if not q.exists():
        raise FileNotFoundError(q)

    return q


def read_manifest(
    path: Path,
) -> List[dict]:

    rows = []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for idx, row in enumerate(reader):
            enroll_raw = first_existing(
                row,
                (
                    "enroll_rel",
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
                    "test_rel",
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

            if (
                enroll_raw is None
                or test_raw is None
                or label_raw is None
            ):
                raise RuntimeError(
                    "Manifest columns "
                    "could not be resolved."
                )

            rows.append(
                {
                    "trial_index": idx,
                    "label":
                        normalize_label(
                            label_raw
                        ),
                    "enroll_path":
                        resolve_audio_path(
                            enroll_raw
                        ),
                    "test_path":
                        resolve_audio_path(
                            test_raw
                        ),
                }
            )

    return rows


# ============================================================
# METRICS
# ============================================================

def verification_metrics(
    labels: List[int],
    scores: List[float],
) -> dict:

    y = np.asarray(
        labels,
        dtype=np.int64,
    )

    s = np.asarray(
        scores,
        dtype=np.float64,
    )

    fpr, tpr, thresholds = roc_curve(
        y,
        s,
        pos_label=1,
    )

    fnr = 1.0 - tpr

    eer_idx = int(
        np.nanargmin(
            np.abs(fnr - fpr)
        )
    )

    eer = (
        float(fpr[eer_idx])
        + float(fnr[eer_idx])
    ) / 2.0

    p_target = 0.01

    c_det = (
        fnr * p_target
        + fpr * (1.0 - p_target)
    )

    dcf_idx = int(
        np.argmin(c_det)
    )

    return {
        "eer_percent":
            100.0 * eer,

        "eer_threshold":
            float(
                thresholds[eer_idx]
            ),

        "min_dcf":
            float(
                c_det[dcf_idx]
                / p_target
            ),

        "min_dcf_threshold":
            float(
                thresholds[dcf_idx]
            ),

        "auc":
            float(
                roc_auc_score(
                    y,
                    s,
                )
            ),
    }


# ============================================================
# TEMPERATURE
# ============================================================

def read_pi_temperature_c():
    candidates = (
        Path(
            "/sys/class/thermal/"
            "thermal_zone0/temp"
        ),
        Path(
            "/sys/devices/virtual/thermal/"
            "thermal_zone0/temp"
        ),
    )

    for path in candidates:
        try:
            value = float(
                path.read_text().strip()
            )

            if value > 1000.0:
                value /= 1000.0

            if -20.0 <= value <= 150.0:
                return float(value)

        except (OSError, ValueError):
            pass

    return None


# ============================================================
# WARM-UP
# ============================================================

def warmup(
    model,
    trials,
):
    unique_files = []
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

            if (
                len(unique_files)
                >= CFG.warmup_files
            ):
                break

        if (
            len(unique_files)
            >= CFG.warmup_files
        ):
            break

    print(
        f"\nWarm-up using "
        f"{len(unique_files)} files..."
    )

    t0 = time.perf_counter()

    for p in unique_files:
        w, _ = audio(
            p,
            CFG.sample_rate,
        )

        _ = emb(
            model,
            w,
        )

    print(
        f"Warm-up complete: "
        f"{time.perf_counter() - t0:.2f} s"
    )


# ============================================================
# PERSISTENT B2 ENROLLMENT CACHE
# OUTSIDE ENERGY BOUNDARY
# ============================================================

def unique_enrollment_keys(
    trials,
):
    keys = []
    seen = set()

    for row in trials:
        key = str(
            row["enroll_path"]
        )

        if key not in seen:
            seen.add(key)
            keys.append(key)

    return keys


def build_cache(
    model,
    trials,
):
    keys = unique_enrollment_keys(
        trials
    )

    print("\n" + "=" * 72)
    print(
        "BUILDING B2 ENROLLMENT "
        "CACHE — NOT TIMED"
    )
    print("=" * 72)
    print(
        f"Unique enrollment "
        f"utterances: {len(keys)}"
    )

    cache = {}

    t0 = time.perf_counter()

    for i, key in enumerate(
        keys,
        start=1,
    ):
        p = Path(key)

        w, _ = audio(
            p,
            CFG.sample_rate,
        )

        cache[key] = emb(
            model,
            w,
        )

        if (
            i == 1
            or i % 25 == 0
            or i == len(keys)
        ):
            print(
                f"  cached "
                f"{i}/{len(keys)}"
            )

    build_s = (
        time.perf_counter() - t0
    )

    payload = {
        "format_version": 1,
        "manifest_sha256":
            CFG.expected_manifest_sha256,
        "b2_checkpoint_sha256":
            CFG.expected_b2_sha256,
        "sample_rate":
            CFG.sample_rate,
        "unique_enrollment_keys":
            keys,
        "cache_b2":
            cache,
    }

    CFG.disk_cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        payload,
        CFG.disk_cache_path,
    )

    print(
        f"Cache build time: "
        f"{build_s:.2f} s — EXCLUDED"
    )

    print(
        f"Persistent cache saved:\n"
        f"  {CFG.disk_cache_path}"
    )

    return cache, {
        "cache_source":
            "rebuilt_from_audio",
        "unique_enrollment_utterances":
            len(keys),
        "cache_build_time_s":
            build_s,
    }


def get_cache(
    model,
    trials,
):
    expected_keys = (
        unique_enrollment_keys(
            trials
        )
    )

    if CFG.disk_cache_path.exists():
        print("\n" + "=" * 72)
        print(
            "CHECKING B2 PERSISTENT "
            "CACHE — NOT TIMED"
        )
        print("=" * 72)

        try:
            t0 = time.perf_counter()

            payload = torch.load(
                CFG.disk_cache_path,
                map_location="cpu",
                weights_only=False,
            )

            checks = {
                "format_version":
                    payload.get(
                        "format_version"
                    ) == 1,

                "manifest_sha256":
                    payload.get(
                        "manifest_sha256"
                    )
                    == CFG.expected_manifest_sha256,

                "b2_checkpoint_sha256":
                    payload.get(
                        "b2_checkpoint_sha256"
                    )
                    == CFG.expected_b2_sha256,

                "sample_rate":
                    int(
                        payload.get(
                            "sample_rate",
                            -1,
                        )
                    )
                    == CFG.sample_rate,

                "enrollment_keys":
                    payload.get(
                        "unique_enrollment_keys"
                    )
                    == expected_keys,
            }

            for name, ok in checks.items():
                print(
                    f"  {name:24s}: "
                    f"{'PASS' if ok else 'FAIL'}"
                )

            if all(checks.values()):
                cache = payload[
                    "cache_b2"
                ]

                if (
                    set(cache.keys())
                    == set(expected_keys)
                ):
                    load_s = (
                        time.perf_counter()
                        - t0
                    )

                    print(
                        "Persistent cache "
                        "integrity: PASS"
                    )

                    print(
                        f"Cache load time: "
                        f"{load_s:.3f} s "
                        f"— EXCLUDED"
                    )

                    return cache, {
                        "cache_source":
                            "persistent_disk_cache",
                        "unique_enrollment_utterances":
                            len(expected_keys),
                        "cache_load_time_s":
                            load_s,
                    }

        except Exception as exc:
            print(
                f"Cache load failed: "
                f"{exc}"
            )

        print(
            "Cache will be rebuilt."
        )

    return build_cache(
        model,
        trials,
    )


# ============================================================
# STATIC-B2 BENCHMARK
# ============================================================

def benchmark_static_b2(
    model,
    cache,
    trials,
):
    print("\n" + "=" * 72)
    print("STATIC_B2")
    print("=" * 72)

    rows = []

    for i, trial in enumerate(
        trials,
        start=1,
    ):
        # EXACT R9 measurement boundary:
        # timer starts BEFORE test audio load.

        t0 = time.perf_counter()

        w_test, duration_s = audio(
            trial["test_path"],
            CFG.sample_rate,
        )

        z_test = emb(
            model,
            w_test,
        )

        score = cosine_score(
            cache[
                str(
                    trial[
                        "enroll_path"
                    ]
                )
            ],
            z_test,
        )

        latency_ms = (
            time.perf_counter()
            - t0
        ) * 1000.0

        rows.append(
            {
                "trial_index":
                    trial["trial_index"],
                "label":
                    trial["label"],
                "enroll_path":
                    str(
                        trial[
                            "enroll_path"
                        ]
                    ),
                "test_path":
                    str(
                        trial[
                            "test_path"
                        ]
                    ),
                "score":
                    score,
                "latency_ms":
                    latency_ms,
                "test_duration_s":
                    duration_s,
                "test_rtf":
                    (
                        latency_ms
                        / 1000.0
                    )
                    / duration_s,
            }
        )

        if (
            i == 1
            or i % 50 == 0
            or i == len(trials)
        ):
            print(
                f"  {i:4d}/"
                f"{len(trials)} | "
                f"lat="
                f"{latency_ms:.2f} ms"
            )

    labels = [
        r["label"]
        for r in rows
    ]

    scores = [
        r["score"]
        for r in rows
    ]

    latencies = np.asarray(
        [
            r["latency_ms"]
            for r in rows
        ],
        dtype=np.float64,
    )

    durations = np.asarray(
        [
            r["test_duration_s"]
            for r in rows
        ],
        dtype=np.float64,
    )

    result = verification_metrics(
        labels,
        scores,
    )

    result.update(
        {
            "mean_latency_ms":
                float(
                    np.mean(
                        latencies
                    )
                ),

            "median_latency_ms":
                float(
                    np.median(
                        latencies
                    )
                ),

            "p95_latency_ms":
                float(
                    np.percentile(
                        latencies,
                        95,
                    )
                ),

            "p99_latency_ms":
                float(
                    np.percentile(
                        latencies,
                        99,
                    )
                ),

            "mean_test_rtf":
                float(
                    np.mean(
                        (
                            latencies
                            / 1000.0
                        )
                        / durations
                    )
                ),
        }
    )

    return rows, result


# ============================================================
# OUTPUT HELPERS
# ============================================================

def read_wh(
    prompt: str,
) -> float:
    while True:
        raw = (
            input(prompt)
            .strip()
            .replace(",", ".")
        )

        try:
            value = float(raw)

        except ValueError:
            print(
                "Enter AT085 Wh "
                "value only."
            )
            continue

        if value < 0:
            print(
                "Wh cannot be negative."
            )
            continue

        return value


def save_csv(
    path: Path,
    rows,
):
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


def append_summary(
    path: Path,
    row: dict,
):
    exists = path.exists()

    with path.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                row.keys()
            ),
        )

        if not exists:
            writer.writeheader()

        writer.writerow(row)


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
    print(
        "R14-C — PI 5 STATIC-b2 / "
        "AT085 SYSTEM-LEVEL ENERGY"
    )
    print("=" * 72)

    print("\nMeasurement boundary:")
    print(
        "  model loading                 : EXCLUDED"
    )
    print(
        "  warm-up                       : EXCLUDED"
    )
    print(
        "  enrollment cache             : EXCLUDED"
    )
    print(
        "  test audio I/O/preprocessing : INCLUDED"
    )
    print(
        "  test embedding inference     : INCLUDED"
    )
    print(
        "  cosine scoring               : INCLUDED"
    )
    print(
        "  cross-trial TEST cache       : DISABLED"
    )
    print(
        "  AT085 measurement            : Pi 5 power input"
    )

    print("\nEnvironment:")
    print(
        f"  Python  : "
        f"{sys.version.split()[0]}"
    )
    print(
        f"  PyTorch : "
        f"{torch.__version__}"
    )
    print(
        f"  machine : "
        f"{platform.machine()}"
    )
    print(
        f"  threads : "
        f"{torch.get_num_threads()}"
    )

    # Manifest
    manifest = discover_manifest()
    manifest_sha = sha256_file(
        manifest
    )

    print("\nManifest:")
    print(
        f"  {manifest}"
    )
    print(
        f"  SHA256: "
        f"{manifest_sha}"
    )

    if (
        manifest_sha
        != CFG.expected_manifest_sha256
    ):
        raise RuntimeError(
            "Manifest SHA gate FAILED."
        )

    trials = read_manifest(
        manifest
    )

    target = sum(
        r["label"] == 1
        for r in trials
    )

    impostor = (
        len(trials) - target
    )

    print(
        f"  trials   : "
        f"{len(trials)}"
    )
    print(
        f"  target   : "
        f"{target}"
    )
    print(
        f"  impostor : "
        f"{impostor}"
    )

    if (
        len(trials)
        != CFG.expected_trials
        or target
        != CFG.expected_target
        or impostor
        != CFG.expected_impostor
    ):
        raise RuntimeError(
            "Trial protocol gate FAILED."
        )

    # b2 checkpoint
    if not CFG.b2_checkpoint.exists():
        raise FileNotFoundError(
            CFG.b2_checkpoint
        )

    b2_sha = sha256_file(
        CFG.b2_checkpoint
    )

    print("\nb2 checkpoint:")
    print(
        f"  {CFG.b2_checkpoint}"
    )
    print(
        f"  SHA256: {b2_sha}"
    )

    if (
        b2_sha
        != CFG.expected_b2_sha256
    ):
        raise RuntimeError(
            "b2 checkpoint SHA gate FAILED."
        )

    print("\nLoading b2...")
    model = load_model(
        CFG.b2_checkpoint
    )

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"b2 params: {params:,}"
    )

    # Outside energy boundary
    warmup(
        model,
        trials,
    )

    cache, cache_info = get_cache(
        model,
        trials,
    )

    print("\n" + "=" * 72)
    print("R14-C PREPARATION COMPLETE")
    print("=" * 72)
    print(
        f"Enrollment cache ready: "
        f"{cache_info['unique_enrollment_utterances']}"
    )
    print(
        f"Cache source: "
        f"{cache_info['cache_source']}"
    )

    summary_csv = (
        CFG.output_dir
        / "r14c_b2_energy_summary.csv"
    )

    # Exactly 3 runs
    for run_number in range(
        1,
        4,
    ):
        print("\n" + "=" * 72)
        print(
            f"R14-C STATIC_B2 "
            f"ENERGY RUN {run_number}/3"
        )
        print("=" * 72)

        print(
            "\nAllow Pi thermal state "
            "to stabilize before this run."
        )

        temp_start = (
            read_pi_temperature_c()
        )

        if temp_start is not None:
            print(
                f"Pi temperature: "
                f"{temp_start:.2f} C"
            )

        print(
            "\nRead accumulated AT085 "
            "energy in Wh."
        )

        start_wh = read_wh(
            "AT085 START Wh = "
        )

        print(
            "\nSTART accepted. "
            "Benchmark starts NOW.\n"
        )

        wall_t0 = (
            time.perf_counter()
        )

        rows, result = (
            benchmark_static_b2(
                model,
                cache,
                trials,
            )
        )

        wall_seconds = (
            time.perf_counter()
            - wall_t0
        )

        temp_end = (
            read_pi_temperature_c()
        )

        print("\n" + "=" * 72)
        print(
            "STATIC_B2 COMPLETE"
        )
        print("=" * 72)

        print(
            f"500-trial wall time: "
            f"{wall_seconds:.3f} s"
        )

        if temp_end is not None:
            print(
                f"Pi temperature end: "
                f"{temp_end:.2f} C"
            )

        print(
            "\nRead AT085 accumulated "
            "Wh NOW."
        )

        end_wh = read_wh(
            "AT085 END Wh = "
        )

        energy_wh = (
            end_wh - start_wh
        )

        if energy_wh <= 0:
            raise RuntimeError(
                "END Wh must be greater "
                "than START Wh."
            )

        energy_j = (
            energy_wh * 3600.0
        )

        j_per_trial = (
            energy_j
            / len(trials)
        )

        avg_power = (
            energy_j
            / wall_seconds
        )

        row = {
            "timestamp":
                time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            "pipeline":
                "STATIC_B2",
            "run_number":
                run_number,
            "trials":
                len(trials),
            "start_wh":
                start_wh,
            "end_wh":
                end_wh,
            "energy_wh":
                energy_wh,
            "energy_j":
                energy_j,
            "energy_j_per_trial":
                j_per_trial,
            "wall_seconds":
                wall_seconds,
            "average_system_power_w":
                avg_power,
            "mean_latency_ms":
                result[
                    "mean_latency_ms"
                ],
            "median_latency_ms":
                result[
                    "median_latency_ms"
                ],
            "p95_latency_ms":
                result[
                    "p95_latency_ms"
                ],
            "p99_latency_ms":
                result[
                    "p99_latency_ms"
                ],
            "eer_percent":
                result[
                    "eer_percent"
                ],
            "min_dcf":
                result[
                    "min_dcf"
                ],
            "auc":
                result[
                    "auc"
                ],
            "pi_temp_start_c":
                temp_start,
            "pi_temp_end_c":
                temp_end,
            "manifest_sha256":
                manifest_sha,
            "b2_checkpoint_sha256":
                b2_sha,
        }

        append_summary(
            summary_csv,
            row,
        )

        trial_csv = (
            CFG.output_dir
            / (
                f"static_b2_run"
                f"{run_number}_trials.csv"
            )
        )

        save_csv(
            trial_csv,
            rows,
        )

        json_path = (
            CFG.output_dir
            / (
                f"static_b2_run"
                f"{run_number}.json"
            )
        )

        with json_path.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                {
                    "experiment":
                        "R14-C",
                    "pipeline":
                        "STATIC_B2",
                    "run_number":
                        run_number,
                    "energy":
                        row,
                    "verification_and_latency":
                        result,
                    "cache":
                        cache_info,
                    "protocol": {
                        "model_loading_energy":
                            "excluded",
                        "warmup_energy":
                            "excluded",
                        "enrollment_cache_energy":
                            "excluded",
                        "test_audio_io_preprocessing_energy":
                            "included",
                        "test_inference_energy":
                            "included",
                        "cosine_scoring_energy":
                            "included",
                        "cross_trial_test_cache":
                            "disabled",
                        "measurement":
                            "system-level energy measured at Raspberry Pi 5 power input",
                    },
                },
                f,
                indent=2,
                default=str,
            )

        print("\nRESULT")
        print(
            f"  Energy         : "
            f"{energy_j:.3f} J"
        )
        print(
            f"  Energy/trial   : "
            f"{j_per_trial:.6f} J"
        )
        print(
            f"  Wall time      : "
            f"{wall_seconds:.3f} s"
        )
        print(
            f"  Average power  : "
            f"{avg_power:.6f} W"
        )

        print(
            f"\nSaved: {json_path}"
        )

        if run_number < 3:
            input(
                "\nAllow the Pi to cool/stabilize. "
                "Press ENTER when ready for "
                "the next run..."
            )

    print("\n" + "=" * 72)
    print("R14-C THREE RUNS COMPLETE")
    print("=" * 72)
    print(
        f"Summary CSV:\n  "
        f"{summary_csv}"
    )
    print(
        "\nDo NOT LOCK until the "
        "three-run results are audited."
    )


if __name__ == "__main__":
    main()