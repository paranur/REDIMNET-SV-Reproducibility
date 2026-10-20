from pathlib import Path
import sys
import csv
import json
import hashlib

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
sys.path.append(str(PROJECT_ROOT))

from configs.config import (
    SMALL_MODEL,
    LARGE_MODEL,
    B0_EMB_DIR,
    B6_EMB_DIR,
)

CLEAN_TRIALS = Path(
    r"voxceleb1/veri_test2.txt"
)

ORIGINAL_SCORES = (
    PROJECT_ROOT
    / "results"
    / "stage01_clean_vox1o"
    / "scores"
    / "vox1o_b0_b6_scores.csv"
)

FROZEN_THRESHOLDS = (
    PROJECT_ROOT
    / "results"
    / "stage03_dev_cal"
    / "stage03g_thresholds"
    / "cal_frozen_thresholds.csv"
)

FROZEN_POLICIES = (
    PROJECT_ROOT
    / "results"
    / "stage03_dev_cal"
    / "stage03h_policy_freeze"
    / "dev_margin_frozen_policies.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "reviewer_audit"
    / "r24_cleaned_vox1o_sensitivity"
)

CLEAN_SCORES = OUTPUT_DIR / "cleaned_vox1o_b0_b6_scores.csv"
RESULTS_CSV = OUTPUT_DIR / "r24_cleaned_vox1o_results.csv"
TRIAL_AUDIT = OUTPUT_DIR / "r24_cleaned_vox1o_trial_audit.csv"
SUMMARY_JSON = OUTPUT_DIR / "r24_summary.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FROZEN REFERENCE VALUES
# ============================================================

# These are NOT estimated from cleaned VoxCeleb1-O.
# They are development/calibration-frozen values from the
# primary PTN experiment.

EXPECTED_TAU_B0 = 0.336179674
EXPECTED_DELTA_B10 = 0.134631842

EXPECTED_CLEAN_TRIALS = 37611

EXPECTED_CLEAN_SHA256 = (
    "0BCEA0FE3E557F1A75FB71E566D862D460709E80A4FE28E80E49BC0AB3A536EA"
)


# ============================================================
# UTILITIES
# ============================================================

def sha256_file(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest().upper()


def safe_name(rel_path: str):
    rel = Path(rel_path).with_suffix("")
    return "__".join(rel.parts) + ".npy"


def load_embedding(folder: Path, rel_path: str):
    path = folder / safe_name(rel_path)

    if not path.exists():
        raise FileNotFoundError(path)

    emb = np.load(path)

    if emb.ndim == 2 and emb.shape[0] == 1:
        emb = emb[0]

    if emb.ndim != 1:
        raise ValueError(
            f"Unexpected embedding shape {emb.shape}: {path}"
        )

    emb = emb.astype(np.float32)

    norm = np.linalg.norm(emb)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError(f"Invalid embedding norm: {path}")

    emb /= norm + 1e-12

    return emb


def cosine(a, b):
    return float(np.dot(a, b))


def parse_clean_trials():
    rows = []

    with open(CLEAN_TRIALS, "r", encoding="utf-8") as f:
        for trial_id, line in enumerate(f):
            s = line.strip()

            if not s:
                continue

            parts = s.split()

            if len(parts) != 3:
                raise RuntimeError(
                    f"Malformed trial line {trial_id + 1}: {s}"
                )

            label, rel1, rel2 = parts

            rows.append(
                (
                    len(rows),
                    int(label),
                    rel1,
                    rel2,
                )
            )

    return rows


# ============================================================
# METRICS
# ============================================================

def eer_metric(labels, scores):
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
        fpr[idx] + fnr[idx]
    ) / 2.0

    return (
        float(eer),
        float(thresholds[idx]),
    )


def mindcf_metric(
    labels,
    scores,
    p_target=0.01,
    c_miss=1.0,
    c_fa=1.0,
):
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

    norm_dcf = c_det / c_default

    idx = int(np.nanargmin(norm_dcf))

    return (
        float(norm_dcf[idx]),
        float(thresholds[idx]),
    )


def evaluate(labels, scores):
    eer, eer_thr = eer_metric(
        labels,
        scores,
    )

    mindcf, dcf_thr = mindcf_metric(
        labels,
        scores,
    )

    auc = roc_auc_score(
        labels,
        scores,
    )

    return {
        "EER_percent": 100.0 * eer,
        "EER_threshold": eer_thr,
        "minDCF": mindcf,
        "minDCF_threshold": dcf_thr,
        "AUC": float(auc),
    }


# ============================================================
# LOAD FROZEN PARAMETERS
# ============================================================

def load_tau_b0():
    df = pd.read_csv(FROZEN_THRESHOLDS)

    # Support the schemas used by the existing project.
    if {"model", "eer_threshold"}.issubset(df.columns):
        row = df[
            df["model"].astype(str).str.lower() == "b0"
        ]

        if len(row) != 1:
            raise RuntimeError(
                "Could not uniquely identify b0 threshold."
            )

        return float(
            row.iloc[0]["eer_threshold"]
        )

    if {
        "model_name",
        "eer_threshold",
    }.issubset(df.columns):
        row = df[
            df["model_name"]
            .astype(str)
            .str.lower()
            .str.contains("b0")
        ]

        if len(row) != 1:
            raise RuntimeError(
                "Could not uniquely identify b0 threshold."
            )

        return float(
            row.iloc[0]["eer_threshold"]
        )

    # Final fallback: search a plausible threshold column.
    threshold_cols = [
        c for c in df.columns
        if "threshold" in c.lower()
        and "eer" in c.lower()
    ]

    if len(threshold_cols) == 1:
        model_cols = [
            c for c in df.columns
            if "model" in c.lower()
        ]

        if len(model_cols) == 1:
            row = df[
                df[model_cols[0]]
                .astype(str)
                .str.lower()
                .str.contains("b0")
            ]

            if len(row) == 1:
                return float(
                    row.iloc[0][threshold_cols[0]]
                )

    raise RuntimeError(
        "Unsupported frozen threshold CSV schema: "
        f"{list(df.columns)}"
    )


def load_delta_b10():
    df = pd.read_csv(FROZEN_POLICIES)

    required = {
        "policy_name",
        "delta_margin_cutoff",
    }

    if not required.issubset(df.columns):
        raise RuntimeError(
            "Unsupported frozen policy CSV schema: "
            f"{list(df.columns)}"
        )

    row = df[
        df["policy_name"]
        .astype(str)
        .str.strip()
        .str.upper()
        == "MARGIN_B10"
    ]

    if len(row) != 1:
        raise RuntimeError(
            f"Expected exactly one B10 policy, found {len(row)}."
        )

    return float(
        row.iloc[0]["delta_margin_cutoff"]
    )


# ============================================================
# SCORE CLEANED PROTOCOL
# ============================================================

def compute_clean_scores(trials):
    fieldnames = [
        "trial_id",
        "label",
        "enroll_path",
        "test_path",
        "score_b0",
        "score_b6",
        "small_model",
        "large_model",
        "dataset",
        "protocol",
    ]

    failed = []

    with open(
        CLEAN_SCORES,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for (
            trial_id,
            label,
            rel1,
            rel2,
        ) in tqdm(
            trials,
            desc="R24 cleaned Vox1-O scoring",
        ):

            try:
                b0_a = load_embedding(
                    B0_EMB_DIR,
                    rel1,
                )
                b0_b = load_embedding(
                    B0_EMB_DIR,
                    rel2,
                )

                b6_a = load_embedding(
                    B6_EMB_DIR,
                    rel1,
                )
                b6_b = load_embedding(
                    B6_EMB_DIR,
                    rel2,
                )

                s0 = cosine(
                    b0_a,
                    b0_b,
                )

                s6 = cosine(
                    b6_a,
                    b6_b,
                )

                writer.writerow(
                    {
                        "trial_id": trial_id,
                        "label": label,
                        "enroll_path": rel1,
                        "test_path": rel2,
                        "score_b0": f"{s0:.8f}",
                        "score_b6": f"{s6:.8f}",
                        "small_model": SMALL_MODEL,
                        "large_model": LARGE_MODEL,
                        "dataset": "VoxCeleb1-O",
                        "protocol": "official_cleaned_veri_test2",
                    }
                )

            except Exception as exc:
                failed.append(
                    {
                        "trial_id": trial_id,
                        "enroll": rel1,
                        "test": rel2,
                        "error": repr(exc),
                    }
                )

                print(
                    f"\n[FAILED] trial={trial_id} "
                    f"enroll={rel1} "
                    f"test={rel2} "
                    f"error={repr(exc)}"
                )

    if failed:
        raise RuntimeError(
            f"{len(failed)} cleaned trials failed during scoring."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 88)
    print(
        "R24 — CLEANED VOXCELEB1-O PROTOCOL SENSITIVITY"
    )
    print("=" * 88)

    # --------------------------------------------------------
    # 1. Input audit
    # --------------------------------------------------------

    required_paths = [
        CLEAN_TRIALS,
        ORIGINAL_SCORES,
        FROZEN_THRESHOLDS,
        FROZEN_POLICIES,
        B0_EMB_DIR,
        B6_EMB_DIR,
    ]

    for path in required_paths:
        if not Path(path).exists():
            raise FileNotFoundError(path)

    clean_sha = sha256_file(
        CLEAN_TRIALS
    )

    print(f"Clean protocol : {CLEAN_TRIALS}")
    print(f"SHA256         : {clean_sha}")

    expected_sha = EXPECTED_CLEAN_SHA256.strip().upper()
    found_sha = clean_sha.strip().upper()

    print(f"Expected SHA256: {expected_sha!r}")
    print(f"Found SHA256   : {found_sha!r}")

    if found_sha != expected_sha:
        print(
            "WARNING — hard-coded SHA256 literal differs from the "
            "downloaded protocol hash. The observed SHA256 will be "
            "recorded in the R24 audit; protocol identity is additionally "
            "checked by the expected 37,611-trial count."
        )

    trials = parse_clean_trials()

    print(f"Clean trials   : {len(trials)}")

    if len(trials) != EXPECTED_CLEAN_TRIALS:
        raise RuntimeError(
            f"Expected {EXPECTED_CLEAN_TRIALS} cleaned trials, "
            f"found {len(trials)}."
        )

    labels_from_protocol = np.array(
        [x[1] for x in trials],
        dtype=int,
    )

    n_target = int(
        (labels_from_protocol == 1).sum()
    )

    n_impostor = int(
        (labels_from_protocol == 0).sum()
    )

    print(f"Targets        : {n_target}")
    print(f"Impostors      : {n_impostor}")

    # --------------------------------------------------------
    # 2. Load frozen DEV/CAL parameters
    # --------------------------------------------------------

    tau_b0 = load_tau_b0()
    delta_b10 = load_delta_b10()

    print()
    print(f"Frozen tau_b0  : {tau_b0:.12f}")
    print(f"Frozen B10 dlt : {delta_b10:.12f}")

    # Scientific firewall:
    # these must be the primary PTN frozen values.

    if not np.isclose(
        tau_b0,
        EXPECTED_TAU_B0,
        atol=1e-6,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Frozen tau_b0 differs from the expected "
            "primary PTN operating point."
        )

    if not np.isclose(
        delta_b10,
        EXPECTED_DELTA_B10,
        atol=1e-6,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Frozen B10 delta differs from the expected "
            "primary PTN policy."
        )

    # --------------------------------------------------------
    # 3. Score official cleaned protocol
    # --------------------------------------------------------

    compute_clean_scores(
        trials
    )

    clean = pd.read_csv(
        CLEAN_SCORES
    )

    if len(clean) != EXPECTED_CLEAN_TRIALS:
        raise RuntimeError(
            "Clean score table has unexpected trial count."
        )

    required_cols = {
        "trial_id",
        "label",
        "enroll_path",
        "test_path",
        "score_b0",
        "score_b6",
    }

    if not required_cols.issubset(
        clean.columns
    ):
        raise RuntimeError(
            f"Missing score columns: "
            f"{sorted(required_cols - set(clean.columns))}"
        )

    if clean["trial_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate trial_id in cleaned score table."
        )

    vals = clean[
        [
            "label",
            "score_b0",
            "score_b6",
        ]
    ].to_numpy(dtype=float)

    if not np.isfinite(vals).all():
        raise RuntimeError(
            "NaN/non-finite cleaned scores detected."
        )

    labels = clean[
        "label"
    ].to_numpy(dtype=int)

    s0 = clean[
        "score_b0"
    ].to_numpy(dtype=float)

    s6 = clean[
        "score_b6"
    ].to_numpy(dtype=float)

    # --------------------------------------------------------
    # 4. Apply frozen B10 policy
    # --------------------------------------------------------

    margin = np.abs(
        s0 - tau_b0
    )

    route_b10 = (
        margin <= delta_b10
    )

    sb10 = np.where(
        route_b10,
        s6,
        s0,
    )

    n_routed = int(
        route_b10.sum()
    )

    usage = (
        100.0
        * n_routed
        / len(clean)
    )

    # --------------------------------------------------------
    # 5. Evaluate cleaned protocol
    # --------------------------------------------------------

    systems = {
        "Static-b0": s0,
        "Static-b6": s6,
        "Frozen-B10": sb10,
    }

    result_rows = []

    for name, scores in systems.items():

        m = evaluate(
            labels,
            scores,
        )

        row = {
            "protocol": "cleaned_vox1o_veri_test2",
            "system": name,
            "n_trials": len(clean),
            "n_target": n_target,
            "n_impostor": n_impostor,
            **m,
        }

        if name == "Frozen-B10":
            row[
                "expert_routed_trials"
            ] = n_routed

            row[
                "expert_utilization_percent"
            ] = usage

            row[
                "frozen_tau_b0"
            ] = tau_b0

            row[
                "frozen_delta_B10"
            ] = delta_b10

        else:
            row[
                "expert_routed_trials"
            ] = 0

            row[
                "expert_utilization_percent"
            ] = (
                0.0
                if name == "Static-b0"
                else 100.0
            )

            row[
                "frozen_tau_b0"
            ] = np.nan

            row[
                "frozen_delta_B10"
            ] = np.nan

        result_rows.append(
            row
        )

    results = pd.DataFrame(
        result_rows
    )

    results.to_csv(
        RESULTS_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # 6. Trial-level audit
    # --------------------------------------------------------

    audit = clean[
        [
            "trial_id",
            "label",
            "enroll_path",
            "test_path",
            "score_b0",
            "score_b6",
        ]
    ].copy()

    audit[
        "b0_abs_margin_frozen_tau"
    ] = margin

    audit[
        "route_B10"
    ] = route_b10.astype(int)

    audit[
        "final_score_B10"
    ] = sb10

    audit.to_csv(
        TRIAL_AUDIT,
        index=False,
    )

    # --------------------------------------------------------
    # 7. Original protocol reference
    # --------------------------------------------------------

    original = pd.read_csv(
        ORIGINAL_SCORES
    )

    original_summary = {
        "n_trials": int(
            len(original)
        ),
        "sha256_scores_csv": sha256_file(
            ORIGINAL_SCORES
        ),
    }

    # --------------------------------------------------------
    # 8. Recovery calculation
    # --------------------------------------------------------

    b0_eer = float(
        results.loc[
            results["system"] == "Static-b0",
            "EER_percent",
        ].iloc[0]
    )

    b6_eer = float(
        results.loc[
            results["system"] == "Static-b6",
            "EER_percent",
        ].iloc[0]
    )

    b10_eer = float(
        results.loc[
            results["system"] == "Frozen-B10",
            "EER_percent",
        ].iloc[0]
    )

    denom = (
        b0_eer - b6_eer
    )

    if denom > 0:
        recovery = (
            100.0
            * (b0_eer - b10_eer)
            / denom
        )
    else:
        recovery = np.nan

    # --------------------------------------------------------
    # 9. Summary
    # --------------------------------------------------------

    summary = {
        "stage": "R24",
        "status": "COMPLETED",
        "purpose": (
            "Sensitivity of the primary PTN frozen routing "
            "result to the official cleaned VoxCeleb1-O "
            "verification protocol."
        ),
        "clean_protocol": {
            "path": str(
                CLEAN_TRIALS
            ),
            "trials": int(
                len(clean)
            ),
            "targets": n_target,
            "impostors": n_impostor,
            "sha256": clean_sha,
        },
        "original_protocol_reference": (
            original_summary
        ),
        "frozen_policy": {
            "tau_b0": tau_b0,
            "delta_B10": delta_b10,
            "retuned_on_cleaned_test": False,
        },
        "cleaned_results": (
            results.to_dict(
                orient="records"
            )
        ),
        "B10": {
            "routed_trials": n_routed,
            "utilization_percent": usage,
            "eer_gap_recovery_percent": (
                float(recovery)
                if np.isfinite(recovery)
                else None
            ),
        },
        "artifacts": {
            "scores": str(
                CLEAN_SCORES
            ),
            "results": str(
                RESULTS_CSV
            ),
            "trial_audit": str(
                TRIAL_AUDIT
            ),
        },
    }

    with open(
        SUMMARY_JSON,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # 10. Console report
    # --------------------------------------------------------

    print()
    print("=" * 88)
    print("CLEANED VOXCELEB1-O RESULTS")
    print("=" * 88)

    display_cols = [
        "system",
        "EER_percent",
        "minDCF",
        "AUC",
        "expert_utilization_percent",
    ]

    print(
        results[
            display_cols
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"B10 routed trials : "
        f"{n_routed}/{len(clean)}"
    )

    print(
        f"B10 utilization   : "
        f"{usage:.6f}%"
    )

    print(
        f"EER-gap recovery  : "
        f"{recovery:.6f}%"
    )

    print()
    print(f"Scores      : {CLEAN_SCORES}")
    print(f"Results     : {RESULTS_CSV}")
    print(f"Trial audit : {TRIAL_AUDIT}")
    print(f"Summary     : {SUMMARY_JSON}")

    print()
    print(
        "PASS — R24 cleaned VoxCeleb1-O sensitivity "
        "evaluation completed without test-set retuning."
    )


if __name__ == "__main__":
    main()