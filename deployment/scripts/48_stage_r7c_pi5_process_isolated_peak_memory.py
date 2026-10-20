from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil
import torch


# ============================================================
# CONFIG
# ============================================================

@dataclass(frozen=True)
class Config:
    project_root: Path = Path(
        "."
    )

    r7a_script: Path = Path(
        "./"
        "scripts/46_stage_r7a_pi5_cached_enrollment_benchmark.py"
    )

    output_dir: Path = Path(
        "./"
        "results/reviewer_audit/"
        "r7c_pi5_process_isolated_peak_memory"
    )

    expected_trials: int = 500
    expected_target: int = 250
    expected_impostor: int = 250

    expected_manifest_sha256: str = (
        "d4e26be792f1ebcb2bf83719d1ee303399cde0dbb299248b6258b4ad436aa2c0"
    )

    tau_b0: float = 0.336179674
    delta_b10: float = 0.134631842

    torch_threads: int = 4
    torch_interop_threads: int = 1

    pipelines: tuple[str, ...] = (
        "FROZEN_B10",
    )


CFG = Config()


# ============================================================
# BASIC HELPERS
# ============================================================

def rss_mb() -> float:
    return (
        psutil.Process(os.getpid()).memory_info().rss
        / (1024.0 ** 2)
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def load_r7a_module():
    if not CFG.r7a_script.exists():
        raise FileNotFoundError(
            f"R7-A script missinexternal/n{CFG.r7a_script}"
        )

    spec = importlib.util.spec_from_file_location(
        "r7a_locked",
        str(CFG.r7a_script),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Could not load R7-A module."
        )

    module = importlib.util.module_from_spec(spec)

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def force_cleanup():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()


# ============================================================
# PEAK RSS SAMPLER
# ============================================================

class PeakRSS:
    """
    Lightweight RSS sampler.

    For this CPU benchmark we explicitly sample RSS after
    important allocation/inference stages and retain the
    largest observed process RSS.

    This is process-level RSS, not tensor-only memory.
    """

    def __init__(self):
        self.peak_mb = rss_mb()

    def sample(self) -> float:
        value = rss_mb()

        if value > self.peak_mb:
            self.peak_mb = value

        return value


# ============================================================
# CACHE BUILDING
# ============================================================

def unique_enrollment_trials(trials: list[dict]) -> list[dict]:
    seen = set()
    out = []

    for trial in trials:
        key = str(trial["enroll_path"])

        if key not in seen:
            seen.add(key)
            out.append(trial)

    return out


def build_cache_for_model(
    r7a,
    model,
    trials: list[dict],
    sampler: PeakRSS,
) -> dict[str, torch.Tensor]:

    unique_trials = unique_enrollment_trials(trials)

    cache = {}

    for i, trial in enumerate(
        unique_trials,
        start=1,
    ):
        key = str(trial["enroll_path"])

        wav, _ = r7a.audio(
            trial["enroll_path"],
            r7a.CFG.sample_rate,
        )

        sampler.sample()

        z = r7a.emb(
            model,
            wav,
        )

        cache[key] = z.detach().cpu()

        sampler.sample()

        del wav
        del z

        if (
            i == 1
            or i % 50 == 0
            or i == len(unique_trials)
        ):
            print(
                f"    cached {i}/{len(unique_trials)} "
                f"| RSS={rss_mb():.2f} MB"
            )

    return cache


# ============================================================
# WARMUP
# ============================================================

def warmup_one_model(
    r7a,
    model,
    trials: list[dict],
    sampler: PeakRSS,
    n: int = 5,
):
    unique_paths = []

    seen = set()

    for trial in trials:
        for key in ("enroll_path", "test_path"):
            p = str(trial[key])

            if p not in seen:
                seen.add(p)
                unique_paths.append(trial[key])

            if len(unique_paths) >= n:
                break

        if len(unique_paths) >= n:
            break

    for p in unique_paths:
        wav, _ = r7a.audio(
            p,
            r7a.CFG.sample_rate,
        )

        _ = r7a.emb(
            model,
            wav,
        )

        sampler.sample()

        del wav
        del _


# ============================================================
# PIPELINE RUNNERS
# ============================================================

def run_static_b0(
    r7a,
    trials,
    sampler,
):
    print("  Loading b0...")

    model_b0 = r7a.load_model(
        r7a.CFG.b0_checkpoint
    )

    sampler.sample()

    print("  Warm-up b0...")
    warmup_one_model(
        r7a,
        model_b0,
        trials,
        sampler,
    )

    print("  Building b0 enrollment cache...")
    cache_b0 = build_cache_for_model(
        r7a,
        model_b0,
        trials,
        sampler,
    )

    cache_ready_rss = sampler.sample()

    print(
        f"  Cache-ready RSS: "
        f"{cache_ready_rss:.2f} MB"
    )

    for i, trial in enumerate(
        trials,
        start=1,
    ):
        wav, _ = r7a.audio(
            trial["test_path"],
            r7a.CFG.sample_rate,
        )

        z0 = r7a.emb(
            model_b0,
            wav,
        )

        _ = r7a.cosine_score(
            cache_b0[str(trial["enroll_path"])],
            z0,
        )

        sampler.sample()

        del wav
        del z0

        if (
            i == 1
            or i % 50 == 0
            or i == len(trials)
        ):
            print(
                f"    {i:4d}/{len(trials)} "
                f"| RSS={rss_mb():.2f} MB "
                f"| peak={sampler.peak_mb:.2f} MB"
            )

    return {
        "pipeline": "STATIC_B0",
        "cache_ready_rss_mb": cache_ready_rss,
        "peak_rss_mb": sampler.peak_mb,
        "routed_count": 0,
        "b6_usage_percent": 0.0,
    }


def run_static_b6(
    r7a,
    trials,
    sampler,
):
    print("  Loading b6...")

    model_b6 = r7a.load_model(
        r7a.CFG.b6_checkpoint
    )

    sampler.sample()

    print("  Warm-up b6...")
    warmup_one_model(
        r7a,
        model_b6,
        trials,
        sampler,
    )

    print("  Building b6 enrollment cache...")
    cache_b6 = build_cache_for_model(
        r7a,
        model_b6,
        trials,
        sampler,
    )

    cache_ready_rss = sampler.sample()

    print(
        f"  Cache-ready RSS: "
        f"{cache_ready_rss:.2f} MB"
    )

    for i, trial in enumerate(
        trials,
        start=1,
    ):
        wav, _ = r7a.audio(
            trial["test_path"],
            r7a.CFG.sample_rate,
        )

        z6 = r7a.emb(
            model_b6,
            wav,
        )

        _ = r7a.cosine_score(
            cache_b6[str(trial["enroll_path"])],
            z6,
        )

        sampler.sample()

        del wav
        del z6

        if (
            i == 1
            or i % 50 == 0
            or i == len(trials)
        ):
            print(
                f"    {i:4d}/{len(trials)} "
                f"| RSS={rss_mb():.2f} MB "
                f"| peak={sampler.peak_mb:.2f} MB"
            )

    return {
        "pipeline": "STATIC_B6",
        "cache_ready_rss_mb": cache_ready_rss,
        "peak_rss_mb": sampler.peak_mb,
        "routed_count": 500,
        "b6_usage_percent": 100.0,
    }


def run_sequential_all(
    r7a,
    trials,
    sampler,
):
    print("  Loading b0...")
    model_b0 = r7a.load_model(
        r7a.CFG.b0_checkpoint
    )
    sampler.sample()

    print("  Loading b6...")
    model_b6 = r7a.load_model(
        r7a.CFG.b6_checkpoint
    )
    sampler.sample()

    print("  Warm-up b0...")
    warmup_one_model(
        r7a,
        model_b0,
        trials,
        sampler,
    )

    print("  Warm-up b6...")
    warmup_one_model(
        r7a,
        model_b6,
        trials,
        sampler,
    )

    print("  Building b0 enrollment cache...")
    cache_b0 = build_cache_for_model(
        r7a,
        model_b0,
        trials,
        sampler,
    )

    print("  Building b6 enrollment cache...")
    cache_b6 = build_cache_for_model(
        r7a,
        model_b6,
        trials,
        sampler,
    )

    cache_ready_rss = sampler.sample()

    print(
        f"  Cache-ready RSS: "
        f"{cache_ready_rss:.2f} MB"
    )

    for i, trial in enumerate(
        trials,
        start=1,
    ):
        # b0 test side
        wav0, _ = r7a.audio(
            trial["test_path"],
            r7a.CFG.sample_rate,
        )

        z0 = r7a.emb(
            model_b0,
            wav0,
        )

        _ = r7a.cosine_score(
            cache_b0[str(trial["enroll_path"])],
            z0,
        )

        sampler.sample()

        del wav0
        del z0

        # Match R7-A: reload test audio for b6.
        wav6, _ = r7a.audio(
            trial["test_path"],
            r7a.CFG.sample_rate,
        )

        z6 = r7a.emb(
            model_b6,
            wav6,
        )

        _ = r7a.cosine_score(
            cache_b6[str(trial["enroll_path"])],
            z6,
        )

        sampler.sample()

        del wav6
        del z6

        if (
            i == 1
            or i % 50 == 0
            or i == len(trials)
        ):
            print(
                f"    {i:4d}/{len(trials)} "
                f"| RSS={rss_mb():.2f} MB "
                f"| peak={sampler.peak_mb:.2f} MB"
            )

    return {
        "pipeline": "SEQUENTIAL_ALL",
        "cache_ready_rss_mb": cache_ready_rss,
        "peak_rss_mb": sampler.peak_mb,
        "routed_count": 500,
        "b6_usage_percent": 100.0,
    }


def run_frozen_b10(
    r7a,
    trials,
    sampler,
):
    print("  Loading b0...")
    model_b0 = r7a.load_model(
        r7a.CFG.b0_checkpoint
    )
    sampler.sample()

    print("  Loading b6...")
    model_b6 = r7a.load_model(
        r7a.CFG.b6_checkpoint
    )
    sampler.sample()

    print("  Warm-up b0...")
    warmup_one_model(
        r7a,
        model_b0,
        trials,
        sampler,
    )

    print("  Warm-up b6...")
    warmup_one_model(
        r7a,
        model_b6,
        trials,
        sampler,
    )

    print("  Building b0 enrollment cache...")
    cache_b0 = build_cache_for_model(
        r7a,
        model_b0,
        trials,
        sampler,
    )

    print("  Building b6 enrollment cache...")
    cache_b6 = build_cache_for_model(
        r7a,
        model_b6,
        trials,
        sampler,
    )

    cache_ready_rss = sampler.sample()

    print(
        f"  Cache-ready RSS: "
        f"{cache_ready_rss:.2f} MB"
    )

    routed_count = 0

    for i, trial in enumerate(
        trials,
        start=1,
    ):
        wav0, _ = r7a.audio(
            trial["test_path"],
            r7a.CFG.sample_rate,
        )

        z0 = r7a.emb(
            model_b0,
            wav0,
        )

        s0 = r7a.cosine_score(
            cache_b0[str(trial["enroll_path"])],
            z0,
        )

        sampler.sample()

        del wav0
        del z0

        routed = (
            abs(
                s0 - CFG.tau_b0
            )
            <= CFG.delta_b10
        )

        if routed:
            routed_count += 1

            wav6, _ = r7a.audio(
                trial["test_path"],
                r7a.CFG.sample_rate,
            )

            z6 = r7a.emb(
                model_b6,
                wav6,
            )

            _ = r7a.cosine_score(
                cache_b6[str(trial["enroll_path"])],
                z6,
            )

            sampler.sample()

            del wav6
            del z6

        if (
            i == 1
            or i % 50 == 0
            or i == len(trials)
        ):
            print(
                f"    {i:4d}/{len(trials)} "
                f"| routed={routed_count} "
                f"| RSS={rss_mb():.2f} MB "
                f"| peak={sampler.peak_mb:.2f} MB"
            )

    usage = (
        100.0
        * routed_count
        / len(trials)
    )

    if routed_count != 40:
        raise RuntimeError(
            "Frozen-B10 integrity gate FAILED: "
            f"expected 40 routed trials, got {routed_count}."
        )

    return {
        "pipeline": "FROZEN_B10",
        "cache_ready_rss_mb": cache_ready_rss,
        "peak_rss_mb": sampler.peak_mb,
        "routed_count": routed_count,
        "b6_usage_percent": usage,
    }


# ============================================================
# CHILD PROCESS
# ============================================================

def child_main(pipeline: str):
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
        f"R7-C CHILD PROCESS — {pipeline}"
    )
    print("=" * 72)

    print(
        f"PID              : {os.getpid()}"
    )
    print(
        f"Initial RSS      : {rss_mb():.2f} MB"
    )

    sampler = PeakRSS()

    r7a = load_r7a_module()

    sampler.sample()

    manifest = r7a.discover_manifest()

    manifest_sha = sha256_file(
        manifest
    )

    print(
        f"Manifest         : {manifest}"
    )
    print(
        f"Manifest SHA256  : {manifest_sha}"
    )

    if (
        manifest_sha
        != CFG.expected_manifest_sha256
    ):
        raise RuntimeError(
            "Manifest SHA256 gate FAILED."
        )

    trials = r7a.read_manifest(
        manifest
    )

    if len(trials) != CFG.expected_trials:
        raise RuntimeError(
            f"Trial-count gate FAILED: "
            f"{len(trials)}"
        )

    target = sum(
        int(t["label"]) == 1
        for t in trials
    )

    impostor = (
        len(trials) - target
    )

    if (
        target != CFG.expected_target
        or impostor != CFG.expected_impostor
    ):
        raise RuntimeError(
            "Target/impostor integrity gate FAILED."
        )

    print(
        f"Trials           : {len(trials)}"
    )
    print(
        f"Target/impostor  : {target}/{impostor}"
    )
    print(
        "Manifest integrity: PASS"
    )

    baseline_rss = sampler.sample()

    print(
        f"Pre-model RSS    : "
        f"{baseline_rss:.2f} MB"
    )

    if pipeline == "STATIC_B0":
        result = run_static_b0(
            r7a,
            trials,
            sampler,
        )

    elif pipeline == "STATIC_B6":
        result = run_static_b6(
            r7a,
            trials,
            sampler,
        )

    elif pipeline == "SEQUENTIAL_ALL":
        result = run_sequential_all(
            r7a,
            trials,
            sampler,
        )

    elif pipeline == "FROZEN_B10":
        result = run_frozen_b10(
            r7a,
            trials,
            sampler,
        )

    else:
        raise ValueError(
            f"Unknown pipeline: {pipeline}"
        )

    result.update(
        {
            "pid": os.getpid(),
            "python":
                sys.version.split()[0],
            "torch":
                torch.__version__,
            "machine":
                platform.machine(),
            "platform":
                platform.platform(),
            "manifest":
                str(manifest),
            "manifest_sha256":
                manifest_sha,
            "trials":
                len(trials),
            "target":
                target,
            "impostor":
                impostor,
            "initial_rss_mb":
                baseline_rss,
            "incremental_peak_above_initial_mb":
                sampler.peak_mb
                - baseline_rss,
            "measurement":
                "fresh-process RSS",
            "enrollment_protocol":
                "cached",
        }
    )

    out = (
        CFG.output_dir
        / f"r7c_{pipeline.lower()}_memory.json"
    )

    with out.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\nRESULT")
    print(
        f"  pipeline       : {pipeline}"
    )
    print(
        f"  initial RSS    : "
        f"{baseline_rss:.2f} MB"
    )
    print(
        f"  cache-ready RSS: "
        f"{result['cache_ready_rss_mb']:.2f} MB"
    )
    print(
        f"  peak RSS       : "
        f"{result['peak_rss_mb']:.2f} MB"
    )
    print(
        f"  incremental    : "
        f"{result['incremental_peak_above_initial_mb']:.2f} MB"
    )
    print(
        f"  routed         : "
        f"{result['routed_count']}"
    )
    print(
        f"  b6 usage       : "
        f"{result['b6_usage_percent']:.6f}%"
    )
    print(
        f"  artifact       : {out}"
    )


# ============================================================
# PARENT PROCESS
# ============================================================

def parent_main():
    CFG.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print(
        "R7-C — PROCESS-ISOLATED PEAK MEMORY BENCHMARK"
    )
    print("=" * 72)

    print(
        "\nEach pipeline runs in a fresh Python process."
    )
    print(
        "Primary metric: process peak RSS."
    )
    print(
        "This experiment does NOT introduce new "
        "accuracy or latency claims."
    )

    child_results = []

    for pipeline in CFG.pipelines:
        print("\n" + "#" * 72)
        print(
            f"STARTING FRESH PROCESS: {pipeline}"
        )
        print("#" * 72)

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            pipeline,
        ]

        t0 = time.perf_counter()

        completed = subprocess.run(
            cmd,
            check=False,
        )

        elapsed = (
            time.perf_counter() - t0
        )

        if completed.returncode != 0:
            raise RuntimeError(
                f"{pipeline} child process failed "
                f"with return code "
                f"{completed.returncode}."
            )

        result_path = (
            CFG.output_dir
            / f"r7c_{pipeline.lower()}_memory.json"
        )

        if not result_path.exists():
            raise RuntimeError(
                f"Missing child artifacexternal/n"
                f"{result_path}"
            )

        with result_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            result = json.load(f)

        result[
            "child_wall_clock_s"
        ] = elapsed

        child_results.append(
            result
        )

        print(
            f"\n{pipeline} child completed "
            f"in {elapsed:.2f} s."
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "experiment":
            "R7-C — Process-Isolated "
            "Peak Memory Benchmark",

        "purpose":
            "Fresh-process peak RSS comparison "
            "for cached-enrollment Pi 5 pipelines.",

        "interpretation":
            (
                "Peak RSS is process-level memory, "
                "including Python, PyTorch, model "
                "weights, runtime workspaces, audio "
                "buffers, and cached embeddings."
            ),

        "config": {
            k: str(v)
            if isinstance(v, Path)
            else v
            for k, v in asdict(CFG).items()
        },

        "results":
            child_results,
    }

    summary_path = (
        CFG.output_dir
        / "r7c_process_isolated_peak_memory_report.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    print("\n")
    print("=" * 72)
    print("R7-C FINAL RESULTS")
    print("=" * 72)

    for r in child_results:
        print(
            f"\n{r['pipeline']}"
        )
        print(
            f"  initial RSS    : "
            f"{r['initial_rss_mb']:.2f} MB"
        )
        print(
            f"  cache-ready RSS: "
            f"{r['cache_ready_rss_mb']:.2f} MB"
        )
        print(
            f"  peak RSS       : "
            f"{r['peak_rss_mb']:.2f} MB"
        )
        print(
            f"  incremental    : "
            f"{r['incremental_peak_above_initial_mb']:.2f} MB"
        )
        print(
            f"  b6 usage       : "
            f"{r['b6_usage_percent']:.2f}%"
        )

    print("\nArtifact:")
    print(
        f"  {summary_path}"
    )

    print("\nR7-C execution complete.")
    print(
        "Do NOT LOCK until output has been audited."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    if (
        len(sys.argv) == 3
        and sys.argv[1] == "--child"
    ):
        child_main(
            sys.argv[2]
        )
    else:
        parent_main()