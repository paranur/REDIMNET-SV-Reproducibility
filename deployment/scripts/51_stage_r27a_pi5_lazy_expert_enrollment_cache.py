# 51_stage_r27a_pi5_lazy_expert_enrollment_cache.py
#
# R27-A — Lazy Expert Enrollment Cache on Raspberry Pi 5
#
# PURPOSE
# -------
# Evaluate a deployment variant of Frozen-B10 in which:
#   1) all b0 enrollment embeddings are provisioned before verification;
#   2) NO b6 enrollment embeddings are initially provisioned;
#   3) when Frozen-B10 routes a trial to b6, the corresponding b6
#      enrollment embedding is created on first use and then cached;
#   4) first-hit b6 enrollment construction is INCLUDED in that trial's
#      measured online latency.
#
# The routing policy, models, manifest, scoring, and trial order are frozen.
# No test-set retuning is permitted.

from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


# ============================================================
# LOAD CANONICAL R7-A IMPLEMENTATION
# ============================================================

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]

R7A_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "46_stage_r7a_pi5_cached_enrollment_benchmark.py"
)

if not R7A_SCRIPT.exists():
    raise FileNotFoundError(R7A_SCRIPT)

spec = importlib.util.spec_from_file_location(
    "r7a_canonical",
    R7A_SCRIPT,
)

if spec is None or spec.loader is None:
    raise RuntimeError(
        f"Cannot import canonical R7-A script: {R7A_SCRIPT}"
    )

r7 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = r7
spec.loader.exec_module(r7)


# ============================================================
# R27-A CONFIG
# ============================================================

OUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "reviewer_audit"
    / "r27a_lazy_expert_enrollment_cache"
)

R7A_REFERENCE = (
    PROJECT_ROOT
    / "results"
    / "reviewer_audit"
    / "r7a_cached_enrollment_pi5_benchmark"
    / "r7a_frozen_b10_trials.csv"
)

TRIAL_OUT = (
    OUT_DIR
    / "r27a_lazy_frozen_b10_trials.csv"
)

SUMMARY_OUT = (
    OUT_DIR
    / "r27a_lazy_expert_cache_report.json"
)

SUMMARY_CSV = (
    OUT_DIR
    / "r27a_lazy_expert_cache_summary.csv"
)

SCORE_TOLERANCE = 1e-6


# ============================================================
# HELPERS
# ============================================================

def cache_tensor_bytes(cache):
    total = 0

    for tensor in cache.values():
        total += (
            tensor.nelement()
            * tensor.element_size()
        )

    return int(total)


def load_r7a_reference():
    if not R7A_REFERENCE.exists():
        raise FileNotFoundError(
            f"R7-A reference missing:\n{R7A_REFERENCE}"
        )

    with R7A_REFERENCE.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        rows = list(csv.DictReader(f))

    if len(rows) != r7.CFG.expected_trials:
        raise RuntimeError(
            "R7-A reference trial-count mismatch: "
            f"{len(rows)}"
        )

    by_index = {}

    for row in rows:
        idx = int(row["trial_index"])

        if idx in by_index:
            raise RuntimeError(
                f"Duplicate R7-A trial index: {idx}"
            )

        by_index[idx] = row

    return by_index


# ============================================================
# INITIAL b0-ONLY ENROLLMENT PROVISIONING
# ============================================================

def build_b0_only_cache(
    model_b0,
    trials,
):
    unique_enroll = []
    seen = set()

    for trial in trials:
        p = trial["enroll_path"]
        key = str(p)

        if key not in seen:
            seen.add(key)
            unique_enroll.append(p)

    print("\n" + "=" * 72)
    print("INITIAL b0-ONLY ENROLLMENT PROVISIONING")
    print("=" * 72)
    print(
        f"Unique enrollment utterances: {len(unique_enroll)}"
    )

    cache_b0 = {}

    rss_before = r7.current_rss_mb()

    t0 = time.perf_counter()

    for i, p in enumerate(
        unique_enroll,
        start=1,
    ):
        w, _ = r7.audio(
            p,
            r7.CFG.sample_rate,
        )

        cache_b0[str(p)] = r7.emb(
            model_b0,
            w,
        )

        if (
            i == 1
            or i % 25 == 0
            or i == len(unique_enroll)
        ):
            print(
                f"  cached b0 {i}/{len(unique_enroll)}"
            )

    elapsed_s = (
        time.perf_counter() - t0
    )

    rss_after = r7.current_rss_mb()

    cache_bytes = cache_tensor_bytes(
        cache_b0
    )

    info = {
        "unique_enrollment_utterances":
            len(unique_enroll),

        "b0_initial_cache_build_time_s":
            float(elapsed_s),

        "b0_initial_mean_build_ms_per_utterance":
            float(
                1000.0
                * elapsed_s
                / len(unique_enroll)
            ),

        "b0_embedding_tensor_bytes":
            cache_bytes,

        "b0_embedding_tensor_mb":
            float(
                cache_bytes
                / (1024.0 ** 2)
            ),

        "rss_before_b0_cache_mb":
            float(rss_before),

        "rss_after_b0_cache_mb":
            float(rss_after),

        "rss_delta_b0_cache_mb":
            float(
                rss_after - rss_before
            ),
    }

    print(
        f"b0 initial provisioning: "
        f"{elapsed_s:.2f} s"
    )

    print(
        f"b0 raw embedding cache: "
        f"{info['b0_embedding_tensor_mb']:.4f} MB"
    )

    return cache_b0, info


# ============================================================
# LAZY FROZEN-B10 TRIAL
# ============================================================

def timed_lazy_frozen_b10(
    model_b0,
    model_b6,
    enroll_path: Path,
    enroll_b0: torch.Tensor,
    lazy_b6_cache,
    test_path: Path,
):
    """
    Timer starts before b0 TEST audio processing.

    If routed:
      - on first use of this enrollment utterance, b6 enrollment audio
        loading + preprocessing + embedding extraction is INCLUDED;
      - the resulting b6 enrollment embedding is cached;
      - b6 TEST audio is then processed as in canonical R7-A.

    No TEST embedding cache is used.
    """

    t0 = time.perf_counter()

    # --------------------------------------------------------
    # Compact test path
    # --------------------------------------------------------

    w0, test_duration_s = r7.audio(
        test_path,
        r7.CFG.sample_rate,
    )

    z0 = r7.emb(
        model_b0,
        w0,
    )

    s0 = r7.cosine_score(
        enroll_b0,
        z0,
    )

    # --------------------------------------------------------
    # Frozen routing
    # --------------------------------------------------------

    route_t0 = time.perf_counter()

    margin = abs(
        s0 - r7.CFG.tau_b0
    )

    routed = bool(
        margin <= r7.CFG.delta_b10
    )

    route_overhead_ms = (
        time.perf_counter()
        - route_t0
    ) * 1000.0

    first_hit = False
    lazy_enroll_build_ms = 0.0

    if routed:
        enroll_key = str(
            enroll_path
        )

        # ----------------------------------------------------
        # Lazy expert enrollment provisioning
        # ----------------------------------------------------

        if enroll_key not in lazy_b6_cache:
            first_hit = True

            lazy_t0 = time.perf_counter()

            w_enroll6, _ = r7.audio(
                enroll_path,
                r7.CFG.sample_rate,
            )

            lazy_b6_cache[
                enroll_key
            ] = r7.emb(
                model_b6,
                w_enroll6,
            )

            lazy_enroll_build_ms = (
                time.perf_counter()
                - lazy_t0
            ) * 1000.0

        enroll_b6 = lazy_b6_cache[
            enroll_key
        ]

        # ----------------------------------------------------
        # Canonical expert TEST path
        # ----------------------------------------------------

        w6, _ = r7.audio(
            test_path,
            r7.CFG.sample_rate,
        )

        z6 = r7.emb(
            model_b6,
            w6,
        )

        s6 = r7.cosine_score(
            enroll_b6,
            z6,
        )

        s_out = s6

    else:
        s6 = float("nan")
        s_out = s0

    latency_ms = (
        time.perf_counter()
        - t0
    ) * 1000.0

    return {
        "score_b0": float(s0),
        "score": float(s_out),
        "routed_b6": int(routed),
        "first_hit_b6_enrollment": int(
            first_hit
        ),
        "lazy_b6_enrollment_build_ms": float(
            lazy_enroll_build_ms
        ),
        "latency_ms": float(
            latency_ms
        ),
        "test_duration_s": float(
            test_duration_s
        ),
        "route_overhead_ms": float(
            route_overhead_ms
        ),
    }


# ============================================================
# LAZY BENCHMARK
# ============================================================

def benchmark_lazy_b10(
    model_b0,
    model_b6,
    cache_b0,
    trials,
):
    print("\n" + "=" * 72)
    print("R27-A — LAZY-EXPERT FROZEN-B10")
    print("=" * 72)

    # Empty by design.
    lazy_b6_cache = {}

    rows = []

    peak_rss = r7.current_rss_mb()

    lazy_rss_before = peak_rss

    for i, trial in enumerate(
        trials,
        start=1,
    ):
        enroll_key = str(
            trial["enroll_path"]
        )

        x = timed_lazy_frozen_b10(
            model_b0=model_b0,
            model_b6=model_b6,
            enroll_path=trial["enroll_path"],
            enroll_b0=cache_b0[
                enroll_key
            ],
            lazy_b6_cache=lazy_b6_cache,
            test_path=trial["test_path"],
        )

        rss = r7.current_rss_mb()
        peak_rss = max(
            peak_rss,
            rss,
        )

        rows.append(
            {
                "trial_index":
                    trial["trial_index"],

                "label":
                    trial["label"],

                "enroll_path":
                    str(
                        trial["enroll_path"]
                    ),

                "test_path":
                    str(
                        trial["test_path"]
                    ),

                "score_b0":
                    x["score_b0"],

                "score":
                    x["score"],

                "routed_b6":
                    x["routed_b6"],

                "first_hit_b6_enrollment":
                    x[
                        "first_hit_b6_enrollment"
                    ],

                "lazy_b6_enrollment_build_ms":
                    x[
                        "lazy_b6_enrollment_build_ms"
                    ],

                "latency_ms":
                    x["latency_ms"],

                "route_overhead_ms":
                    x["route_overhead_ms"],

                "test_duration_s":
                    x["test_duration_s"],

                "test_rtf":
                    (
                        x["latency_ms"]
                        / 1000.0
                    )
                    / x["test_duration_s"],

                "lazy_b6_cache_size_after_trial":
                    len(lazy_b6_cache),
            }
        )

        if (
            i == 1
            or i % 50 == 0
            or i == len(trials)
        ):
            print(
                f"  {i:4d}/{len(trials)} | "
                f"route={x['routed_b6']} | "
                f"first_hit="
                f"{x['first_hit_b6_enrollment']} | "
                f"b6_cache="
                f"{len(lazy_b6_cache)} | "
                f"lat={x['latency_ms']:.2f} ms"
            )

    # --------------------------------------------------------
    # Main verification / latency summary
    # --------------------------------------------------------

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

    routed_rows = [
        r
        for r in rows
        if r["routed_b6"] == 1
    ]

    first_hit_rows = [
        r
        for r in routed_rows
        if r["first_hit_b6_enrollment"] == 1
    ]

    cached_hit_rows = [
        r
        for r in routed_rows
        if r["first_hit_b6_enrollment"] == 0
    ]

    nonrouted_rows = [
        r
        for r in rows
        if r["routed_b6"] == 0
    ]

    result = {}

    result.update(
        r7.verification_metrics(
            labels,
            scores,
        )
    )

    result.update(
        r7.latency_summary(
            latencies,
            durations,
        )
    )

    result["max_rss_mb"] = float(
        peak_rss
    )

    result["routed_count"] = int(
        len(routed_rows)
    )

    result["b6_usage_percent"] = float(
        100.0
        * len(routed_rows)
        / len(rows)
    )

    result["mean_route_overhead_ms"] = float(
        np.mean(route_overheads)
    )

    result["p95_route_overhead_ms"] = float(
        np.percentile(
            route_overheads,
            95,
        )
    )

    # --------------------------------------------------------
    # Lazy expert-cache summary
    # --------------------------------------------------------

    lazy_build_times = [
        r["lazy_b6_enrollment_build_ms"]
        for r in first_hit_rows
    ]

    lazy_build_total_ms = float(
        np.sum(lazy_build_times)
    )

    b6_cache_bytes = cache_tensor_bytes(
        lazy_b6_cache
    )

    lazy_rss_after = r7.current_rss_mb()

    lazy_info = {
        "initial_b6_cache_entries": 0,

        "final_b6_cache_entries":
            int(len(lazy_b6_cache)),

        "unique_b6_enrollments_materialized":
            int(len(lazy_b6_cache)),

        "expert_enrollment_coverage_percent":
            float(
                100.0
                * len(lazy_b6_cache)
                / len(cache_b0)
            ),

        "routed_trials":
            int(len(routed_rows)),

        "first_hit_routed_trials":
            int(len(first_hit_rows)),

        "cached_hit_routed_trials":
            int(len(cached_hit_rows)),

        "lazy_b6_build_total_s":
            float(
                lazy_build_total_ms
                / 1000.0
            ),

        "lazy_b6_build_mean_ms_per_materialized_enrollment":
            (
                float(
                    np.mean(
                        lazy_build_times
                    )
                )
                if lazy_build_times
                else 0.0
            ),

        "lazy_b6_build_median_ms":
            (
                float(
                    np.median(
                        lazy_build_times
                    )
                )
                if lazy_build_times
                else 0.0
            ),

        "lazy_b6_build_p95_ms":
            (
                float(
                    np.percentile(
                        lazy_build_times,
                        95,
                    )
                )
                if lazy_build_times
                else 0.0
            ),

        "lazy_b6_embedding_tensor_bytes":
            int(b6_cache_bytes),

        "lazy_b6_embedding_tensor_mb":
            float(
                b6_cache_bytes
                / (1024.0 ** 2)
            ),

        "rss_before_lazy_b6_materialization_mb":
            float(
                lazy_rss_before
            ),

        "rss_after_lazy_b6_materialization_mb":
            float(
                lazy_rss_after
            ),

        "rss_delta_lazy_phase_mb":
            float(
                lazy_rss_after
                - lazy_rss_before
            ),
    }

    def latency_group(group):
        if not group:
            return {
                "count": 0,
                "mean_latency_ms": None,
                "median_latency_ms": None,
                "p95_latency_ms": None,
                "p99_latency_ms": None,
            }

        x = np.asarray(
            [
                r["latency_ms"]
                for r in group
            ],
            dtype=np.float64,
        )

        return {
            "count": int(len(group)),
            "mean_latency_ms":
                float(np.mean(x)),
            "median_latency_ms":
                float(np.median(x)),
            "p95_latency_ms":
                float(np.percentile(x, 95)),
            "p99_latency_ms":
                float(np.percentile(x, 99)),
        }

    latency_groups = {
        "non_routed":
            latency_group(
                nonrouted_rows
            ),

        "routed_all":
            latency_group(
                routed_rows
            ),

        "routed_first_hit":
            latency_group(
                first_hit_rows
            ),

        "routed_cached_hit":
            latency_group(
                cached_hit_rows
            ),
    }

    return (
        rows,
        result,
        lazy_info,
        latency_groups,
    )


# ============================================================
# R7-A SCORE INTEGRITY
# ============================================================

def compare_with_r7a(
    rows,
    reference_by_index,
):
    max_abs_final = 0.0
    max_abs_b0 = 0.0

    route_mismatch = 0
    label_mismatch = 0
    path_mismatch = 0

    for row in rows:
        idx = int(
            row["trial_index"]
        )

        if idx not in reference_by_index:
            raise RuntimeError(
                f"Trial {idx} missing from R7-A reference."
            )

        ref = reference_by_index[
            idx
        ]

        if int(row["label"]) != int(
            ref["label"]
        ):
            label_mismatch += 1

        if (
            str(row["enroll_path"])
            != str(ref["enroll_path"])
            or str(row["test_path"])
            != str(ref["test_path"])
        ):
            path_mismatch += 1

        route_mismatch += int(
            int(row["routed_b6"])
            != int(ref["routed_b6"])
        )

        max_abs_b0 = max(
            max_abs_b0,
            abs(
                float(row["score_b0"])
                - float(ref["score_b0"])
            ),
        )

        max_abs_final = max(
            max_abs_final,
            abs(
                float(row["score"])
                - float(ref["score"])
            ),
        )

    passed = bool(
        max_abs_b0 <= SCORE_TOLERANCE
        and max_abs_final <= SCORE_TOLERANCE
        and route_mismatch == 0
        and label_mismatch == 0
        and path_mismatch == 0
    )

    return {
        "reference":
            str(R7A_REFERENCE),

        "tolerance":
            SCORE_TOLERANCE,

        "max_abs_b0_score_difference":
            float(max_abs_b0),

        "max_abs_final_score_difference":
            float(max_abs_final),

        "routing_mismatches":
            int(route_mismatch),

        "label_mismatches":
            int(label_mismatch),

        "path_mismatches":
            int(path_mismatch),

        "pass":
            passed,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.set_num_threads(
        r7.CFG.torch_threads
    )

    try:
        torch.set_num_interop_threads(
            r7.CFG.torch_interop_threads
        )
    except RuntimeError:
        pass

    print("=" * 72)
    print(
        "R27-A — LAZY EXPERT ENROLLMENT CACHE "
        "RASPBERRY PI 5"
    )
    print("=" * 72)

    print("\nProtocol:")
    print(
        "  initial enrollment provisioning : "
        "b0 only, outside online timing"
    )
    print(
        "  initial b6 enrollment cache      : EMPTY"
    )
    print(
        "  first routed b6 enrollment build : TIMED"
    )
    print(
        "  subsequent b6 enrollment reuse   : CACHED"
    )
    print(
        "  b0 test inference                : TIMED"
    )
    print(
        "  b6 test inference if routed      : TIMED"
    )
    print(
        "  cross-trial TEST cache           : DISABLED"
    )

    print("\nFrozen policy:")
    print(
        f"  tau_b0    = "
        f"{r7.CFG.tau_b0:.9f}"
    )
    print(
        f"  delta_B10 = "
        f"{r7.CFG.delta_b10:.9f}"
    )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest = r7.discover_manifest()

    manifest_sha = r7.sha256_file(
        manifest
    )

    if (
        manifest_sha
        != r7.CFG.expected_manifest_sha256
    ):
        raise RuntimeError(
            "Manifest SHA256 gate FAILED."
        )

    trials = r7.read_manifest(
        manifest
    )

    n_target = sum(
        int(x["label"] == 1)
        for x in trials
    )

    n_impostor = sum(
        int(x["label"] == 0)
        for x in trials
    )

    if (
        len(trials)
        != r7.CFG.expected_trials
        or n_target
        != r7.CFG.expected_target
        or n_impostor
        != r7.CFG.expected_impostor
    ):
        raise RuntimeError(
            "Canonical trial protocol gate FAILED."
        )

    print("\nManifest:")
    print(" ", manifest)
    print("  SHA256 :", manifest_sha)
    print("  trials :", len(trials))
    print("  target :", n_target)
    print("  impostor:", n_impostor)

    # --------------------------------------------------------
    # Reference
    # --------------------------------------------------------

    reference = load_r7a_reference()

    print(
        "\nR7-A reference loaded:",
        len(reference),
        "trials",
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    print("\nLoading b0...")
    model_b0 = r7.load_model(
        r7.CFG.b0_checkpoint
    )

    print("Loading b6...")
    model_b6 = r7.load_model(
        r7.CFG.b6_checkpoint
    )

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    r7.warmup_models(
        model_b0,
        model_b6,
        trials,
    )

    # --------------------------------------------------------
    # Initial provisioning — b0 only
    # --------------------------------------------------------

    cache_b0, b0_provisioning = (
        build_b0_only_cache(
            model_b0,
            trials,
        )
    )

    # --------------------------------------------------------
    # Lazy B10 benchmark
    # --------------------------------------------------------

    (
        rows,
        result,
        lazy_info,
        latency_groups,
    ) = benchmark_lazy_b10(
        model_b0,
        model_b6,
        cache_b0,
        trials,
    )

    # --------------------------------------------------------
    # Scientific integrity
    # --------------------------------------------------------

    integrity = compare_with_r7a(
        rows,
        reference,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    r7.save_csv(
        TRIAL_OUT,
        rows,
    )

    summary_row = {
        "trials":
            len(rows),

        "routed_trials":
            result["routed_count"],

        "b6_usage_percent":
            result["b6_usage_percent"],

        "initial_b0_cache_build_s":
            b0_provisioning[
                "b0_initial_cache_build_time_s"
            ],

        "lazy_b6_unique_enrollments":
            lazy_info[
                "unique_b6_enrollments_materialized"
            ],

        "expert_enrollment_coverage_percent":
            lazy_info[
                "expert_enrollment_coverage_percent"
            ],

        "lazy_b6_build_total_s":
            lazy_info[
                "lazy_b6_build_total_s"
            ],

        "eer_percent":
            result["eer_percent"],

        "min_dcf":
            result["min_dcf"],

        "auc":
            result["auc"],

        "mean_latency_ms":
            result["mean_latency_ms"],

        "median_latency_ms":
            result["median_latency_ms"],

        "p95_latency_ms":
            result["p95_latency_ms"],

        "p99_latency_ms":
            result["p99_latency_ms"],

        "score_integrity_pass":
            int(integrity["pass"]),
    }

    r7.save_csv(
        SUMMARY_CSV,
        [summary_row],
    )

    report = {
        "experiment":
            "R27-A — Lazy Expert Enrollment Cache",

        "status":
            "COMPLETE_PENDING_SCIENTIFIC_LOCK",

        "protocol": {
            "initial_enrollment_cache":
                "b0_only",

            "initial_b6_cache":
                "empty",

            "lazy_b6_enrollment_materialization":
                "first_routed_use",

            "first_hit_b6_enrollment_cost":
                "included_in_online_trial_latency",

            "subsequent_b6_enrollment_use":
                "cached",

            "test_embedding_cache":
                "disabled",

            "routing_policy":
                "frozen_unchanged",

            "trial_order":
                "canonical_manifest_order",
        },

        "manifest": {
            "path":
                str(manifest),
            "sha256":
                manifest_sha,
            "trials":
                len(trials),
            "target":
                n_target,
            "impostor":
                n_impostor,
        },

        "frozen_policy": {
            "tau_b0":
                r7.CFG.tau_b0,
            "delta_b10":
                r7.CFG.delta_b10,
        },

        "environment": {
            "python":
                sys.version,
            "torch":
                torch.__version__,
            "machine":
                r7.platform.machine(),
            "platform":
                r7.platform.platform(),
            "cpu_count":
                os.cpu_count(),
            "torch_threads":
                torch.get_num_threads(),
            "torch_interop_threads":
                torch.get_num_interop_threads(),
        },

        "checkpoints": {
            "b0":
                str(
                    r7.CFG.b0_checkpoint
                ),
            "b6":
                str(
                    r7.CFG.b6_checkpoint
                ),
            "b0_sha256":
                r7.sha256_file(
                    r7.CFG.b0_checkpoint
                ),
            "b6_sha256":
                r7.sha256_file(
                    r7.CFG.b6_checkpoint
                ),
        },

        "initial_b0_provisioning":
            b0_provisioning,

        "lazy_b6_materialization":
            lazy_info,

        "verification_and_latency":
            result,

        "latency_groups":
            latency_groups,

        "r7a_score_integrity":
            integrity,
    }

    with SUMMARY_OUT.open(
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
    # Console summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 72)
    print("R27-A FINAL OUTPUT")
    print("=" * 72)

    print("\nINITIAL b0 PROVISIONING")
    print(
        f"  utterances : "
        f"{b0_provisioning['unique_enrollment_utterances']}"
    )
    print(
        f"  time       : "
        f"{b0_provisioning['b0_initial_cache_build_time_s']:.2f} s"
    )
    print(
        f"  mean/utt   : "
        f"{b0_provisioning['b0_initial_mean_build_ms_per_utterance']:.2f} ms"
    )
    print(
        f"  cache      : "
        f"{b0_provisioning['b0_embedding_tensor_mb']:.4f} MB"
    )

    print("\nLAZY b6 MATERIALIZATION")
    print(
        f"  routed trials       : "
        f"{result['routed_count']}/{len(rows)}"
    )
    print(
        f"  b6 usage            : "
        f"{result['b6_usage_percent']:.6f}%"
    )
    print(
        f"  unique b6 enrollments: "
        f"{lazy_info['unique_b6_enrollments_materialized']}"
    )
    print(
        f"  coverage            : "
        f"{lazy_info['expert_enrollment_coverage_percent']:.3f}%"
    )
    print(
        f"  total b6 build      : "
        f"{lazy_info['lazy_b6_build_total_s']:.2f} s"
    )

    print("\nVERIFICATION")
    print(
        f"  EER      : "
        f"{result['eer_percent']:.6f}%"
    )
    print(
        f"  minDCF   : "
        f"{result['min_dcf']:.9f}"
    )
    print(
        f"  AUC      : "
        f"{result['auc']:.9f}"
    )

    print("\nONLINE LATENCY — FIRST-HIT COST INCLUDED")
    print(
        f"  mean     : "
        f"{result['mean_latency_ms']:.2f} ms"
    )
    print(
        f"  median   : "
        f"{result['median_latency_ms']:.2f} ms"
    )
    print(
        f"  P95      : "
        f"{result['p95_latency_ms']:.2f} ms"
    )
    print(
        f"  P99      : "
        f"{result['p99_latency_ms']:.2f} ms"
    )

    for name, x in latency_groups.items():
        print(
            f"\n  {name}: "
            f"n={x['count']} | "
            f"mean={x['mean_latency_ms']} | "
            f"median={x['median_latency_ms']} | "
            f"P95={x['p95_latency_ms']} | "
            f"P99={x['p99_latency_ms']}"
        )

    print("\nR7-A SCORE / ROUTING INTEGRITY")
    print(
        f"  max |Δ b0|    : "
        f"{integrity['max_abs_b0_score_difference']:.3e}"
    )
    print(
        f"  max |Δ final| : "
        f"{integrity['max_abs_final_score_difference']:.3e}"
    )
    print(
        f"  route mismatch: "
        f"{integrity['routing_mismatches']}"
    )
    print(
        "  STATUS        : "
        + (
            "PASS"
            if integrity["pass"]
            else "FAIL"
        )
    )

    print("\nArtifacts:")
    print(" ", TRIAL_OUT)
    print(" ", SUMMARY_CSV)
    print(" ", SUMMARY_OUT)

    if not integrity["pass"]:
        raise RuntimeError(
            "R27-A FAILED R7-A scientific-integrity gate."
        )

    print(
        "\nCOMPLETE — do not scientifically LOCK "
        "until results are reviewed."
    )


if __name__ == "__main__":
    main()