"""
ReDimNet-SV MASTER
Stage R2-B — Frozen Mixed-Stream Decision Threshold Transfer Audit

Question
--------
For Original Frozen-B10 mixed scores, what happens when the FINAL
accept/reject decision threshold is selected on CAL and transferred
unchanged to held-out VoxCeleb1-O?

IMPORTANT
---------
- Original Frozen-B10 routing is NOT modified.
- CAL selects/freezes decision thresholds.
- VoxCeleb1-O does NOT select a threshold.
- No test-set threshold sweep is used to make the fixed decisions.
- No Platt/logistic calibration is fitted in this stage.

Two operating points:
1) CAL EER-oriented threshold.
2) CAL normalized-minDCF-oriented threshold, Ptarget=0.01.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


@dataclass(frozen=True)
class Config:
    root: str = r"package/"

    cal_rel: str = (
        r"results\stage03_dev_cal\stage03f_scores"
        r"\cal_b0_b6_scores_20k.csv"
    )

    test_rel: str = (
        r"results\stage01_clean_vox1o\scores"
        r"\vox1o_b0_b6_scores.csv"
    )

    cal_sha256: str = (
        "d48855e7c95a58a52c79d315afac5332fb30d9484fdbffdfa73783343c246cdf"
    )

    test_sha256: str = (
        "f71d954a7e237f8d43140f03df132d88cabb9e6cbdf5659fd92c94055d38ee4c"
    )

    cal_rows: int = 20_000
    test_rows: int = 37_720

    # Original Frozen-B10 routing policy.
    tau_b0: float = 0.336179674
    delta_b10: float = 0.134631842

    p_target: float = 0.01
    c_miss: float = 1.0
    c_fa: float = 1.0

    output_rel: str = (
        r"results\reviewer_audit"
        r"\r2b_fixed_decision_threshold_transfer"
    )


CFG = Config()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_scores(
    path: Path,
    expected_hash: str,
    expected_rows: int,
    name: str,
):
    if not path.exists():
        raise FileNotFoundError(path)

    actual_hash = sha256_file(path)

    print(f"{name} file   : {path}")
    print(f"{name} SHA256 : {actual_hash}")

    if actual_hash != expected_hash:
        raise RuntimeError(
            f"{name} SHA256 mismatch."
        )

    df = pd.read_csv(path)

    if len(df) != expected_rows:
        raise RuntimeError(
            f"{name}: expected {expected_rows} rows, "
            f"got {len(df)}"
        )

    required = {
        "trial_id",
        "label",
        "score_b0",
        "score_b6",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"{name}: missing columns {sorted(missing)}"
        )

    labels = df["label"].astype(int).to_numpy()
    b0 = df["score_b0"].astype(float).to_numpy()
    b6 = df["score_b6"].astype(float).to_numpy()

    if not np.isfinite(b0).all():
        raise RuntimeError(f"{name}: non-finite b0.")

    if not np.isfinite(b6).all():
        raise RuntimeError(f"{name}: non-finite b6.")

    print(f"{name} rows    : {len(df)}")
    print(f"{name} target  : {int(labels.sum())}")
    print(f"{name} impostor: {int((labels == 0).sum())}")
    print(f"{name} integrity: PASS")
    print()

    return df, labels, b0, b6


def make_original_b10(b0, b6):
    route = (
        np.abs(b0 - CFG.tau_b0)
        <= CFG.delta_b10
    )

    mixed = np.where(
        route,
        b6,
        b0,
    )

    return route, mixed


def find_eer_threshold(labels, scores):
    fpr, tpr, thresholds = roc_curve(
        labels,
        scores,
        pos_label=1,
    )

    fnr = 1.0 - tpr

    idx = int(
        np.argmin(
            np.abs(fnr - fpr)
        )
    )

    return {
        "threshold": float(thresholds[idx]),
        "FPR": float(fpr[idx]),
        "FNR": float(fnr[idx]),
        "EER_percent":
            float(
                0.5
                * (fpr[idx] + fnr[idx])
                * 100.0
            ),
    }


def find_mindcf_threshold(labels, scores):
    fpr, tpr, thresholds = roc_curve(
        labels,
        scores,
        pos_label=1,
    )

    fnr = 1.0 - tpr

    c_det = (
        CFG.c_miss * fnr * CFG.p_target
        +
        CFG.c_fa * fpr * (1.0 - CFG.p_target)
    )

    c_default = min(
        CFG.c_miss * CFG.p_target,
        CFG.c_fa * (1.0 - CFG.p_target),
    )

    norm = c_det / c_default

    idx = int(np.argmin(norm))

    return {
        "threshold": float(thresholds[idx]),
        "FPR": float(fpr[idx]),
        "FNR": float(fnr[idx]),
        "normalized_DCF": float(norm[idx]),
    }


def evaluate_fixed_threshold(labels, scores, threshold):
    """
    Accept target iff score >= threshold.
    """

    pred_target = scores >= threshold

    target = labels == 1
    impostor = labels == 0

    misses = int(
        np.sum(
            target & (~pred_target)
        )
    )

    false_accepts = int(
        np.sum(
            impostor & pred_target
        )
    )

    target_n = int(target.sum())
    impostor_n = int(impostor.sum())

    fnr = misses / target_n
    fpr = false_accepts / impostor_n

    far_percent = 100.0 * fpr
    frr_percent = 100.0 * fnr

    balanced_error_percent = (
        100.0
        * 0.5
        * (fpr + fnr)
    )

    raw_cost = (
        CFG.c_miss
        * fnr
        * CFG.p_target
        +
        CFG.c_fa
        * fpr
        * (1.0 - CFG.p_target)
    )

    c_default = min(
        CFG.c_miss * CFG.p_target,
        CFG.c_fa * (1.0 - CFG.p_target),
    )

    normalized_dcf = raw_cost / c_default

    normalized_miss_component = (
        CFG.c_miss
        * fnr
        * CFG.p_target
        / c_default
    )

    normalized_fa_component = (
        CFG.c_fa
        * fpr
        * (1.0 - CFG.p_target)
        / c_default
    )

    return {
        "threshold": float(threshold),

        "target_trials": target_n,
        "impostor_trials": impostor_n,

        "misses": misses,
        "false_accepts": false_accepts,

        "FNR": float(fnr),
        "FPR": float(fpr),

        "FRR_percent": float(frr_percent),
        "FAR_percent": float(far_percent),

        "balanced_error_percent":
            float(balanced_error_percent),

        "normalized_DCF":
            float(normalized_dcf),

        "normalized_miss_component":
            float(normalized_miss_component),

        "normalized_fa_component":
            float(normalized_fa_component),
    }


def main():

    print("=" * 96)
    print(
        "STAGE R2-B — FROZEN MIXED-STREAM "
        "DECISION-THRESHOLD TRANSFER AUDIT"
    )
    print("=" * 96)

    print()
    print("CAL freezes the final decision thresholds.")
    print("VoxCeleb1-O evaluates them unchanged.")
    print("Original Frozen-B10 routing is unchanged.")
    print("NO test-set threshold selection.")
    print("NO calibration model fitting.")
    print()

    root = Path(CFG.root)

    _, cal_y, cal_b0, cal_b6 = load_scores(
        root / CFG.cal_rel,
        CFG.cal_sha256,
        CFG.cal_rows,
        "CAL",
    )

    _, test_y, test_b0, test_b6 = load_scores(
        root / CFG.test_rel,
        CFG.test_sha256,
        CFG.test_rows,
        "VOX1O",
    )

    # --------------------------------------------------------
    # Original Frozen-B10 mixed streams
    # --------------------------------------------------------

    cal_route, cal_mixed = make_original_b10(
        cal_b0,
        cal_b6,
    )

    test_route, test_mixed = make_original_b10(
        test_b0,
        test_b6,
    )

    print("=" * 96)
    print("ORIGINAL FROZEN-B10 ROUTING")
    print("=" * 96)

    print(
        f"CAL b6 usage  : "
        f"{100.0 * cal_route.mean():.6f}%"
    )

    print(
        f"VOX1O b6 usage: "
        f"{100.0 * test_route.mean():.6f}%"
    )

    print()

    # --------------------------------------------------------
    # CAL ONLY: freeze final operating thresholds
    # --------------------------------------------------------

    cal_eer_op = find_eer_threshold(
        cal_y,
        cal_mixed,
    )

    cal_dcf_op = find_mindcf_threshold(
        cal_y,
        cal_mixed,
    )

    theta_eer = cal_eer_op["threshold"]
    theta_dcf = cal_dcf_op["threshold"]

    print("=" * 96)
    print("CAL-FROZEN FINAL DECISION THRESHOLDS")
    print("=" * 96)

    print(
        f"theta_mixed_EER = "
        f"{theta_eer:.12f}"
    )

    print(
        f"CAL EER         = "
        f"{cal_eer_op['EER_percent']:.6f}%"
    )

    print()

    print(
        f"theta_mixed_DCF = "
        f"{theta_dcf:.12f}"
    )

    print(
        f"CAL minDCF      = "
        f"{cal_dcf_op['normalized_DCF']:.9f}"
    )

    print()

    # --------------------------------------------------------
    # Fixed threshold evaluation on CAL itself
    # sanity/reproduction
    # --------------------------------------------------------

    cal_at_eer = evaluate_fixed_threshold(
        cal_y,
        cal_mixed,
        theta_eer,
    )

    cal_at_dcf = evaluate_fixed_threshold(
        cal_y,
        cal_mixed,
        theta_dcf,
    )

    # --------------------------------------------------------
    # HELD-OUT TRANSFER
    # NO threshold sweep here.
    # --------------------------------------------------------

    test_at_eer = evaluate_fixed_threshold(
        test_y,
        test_mixed,
        theta_eer,
    )

    test_at_dcf = evaluate_fixed_threshold(
        test_y,
        test_mixed,
        theta_dcf,
    )

    rows = []

    for split, operating_point, result in [
        (
            "CAL",
            "CAL-EER-threshold",
            cal_at_eer,
        ),
        (
            "CAL",
            "CAL-minDCF-threshold",
            cal_at_dcf,
        ),
        (
            "VOX1O",
            "CAL-EER-threshold",
            test_at_eer,
        ),
        (
            "VOX1O",
            "CAL-minDCF-threshold",
            test_at_dcf,
        ),
    ]:
        rows.append({
            "split": split,
            "operating_point": operating_point,
            **result,
        })

    result_df = pd.DataFrame(rows)

    print("=" * 96)
    print("FIXED-THRESHOLD TRANSFER RESULTS")
    print("=" * 96)

    print(
        result_df[
            [
                "split",
                "operating_point",
                "threshold",
                "misses",
                "false_accepts",
                "FAR_percent",
                "FRR_percent",
                "balanced_error_percent",
                "normalized_DCF",
                "normalized_miss_component",
                "normalized_fa_component",
            ]
        ].to_string(index=False)
    )

    print()

    # --------------------------------------------------------
    # Compare held-out fixed thresholds with test-set oracle
    # operating points ONLY as diagnostic references.
    #
    # IMPORTANT:
    # These oracle thresholds are not used for deployment decisions.
    # --------------------------------------------------------

    test_oracle_eer = find_eer_threshold(
        test_y,
        test_mixed,
    )

    test_oracle_dcf = find_mindcf_threshold(
        test_y,
        test_mixed,
    )

    print("=" * 96)
    print("HELD-OUT ORACLE REFERENCES — DIAGNOSTIC ONLY")
    print("=" * 96)

    print(
        f"VOX1O oracle EER threshold : "
        f"{test_oracle_eer['threshold']:.12f}"
    )

    print(
        f"VOX1O oracle EER           : "
        f"{test_oracle_eer['EER_percent']:.6f}%"
    )

    print(
        f"CAL -> VOX1O EER-threshold shift: "
        f"{theta_eer - test_oracle_eer['threshold']:+.12f}"
    )

    print()

    print(
        f"VOX1O oracle minDCF threshold: "
        f"{test_oracle_dcf['threshold']:.12f}"
    )

    print(
        f"VOX1O oracle minDCF          : "
        f"{test_oracle_dcf['normalized_DCF']:.9f}"
    )

    print(
        f"CAL -> VOX1O DCF-threshold shift: "
        f"{theta_dcf - test_oracle_dcf['threshold']:+.12f}"
    )

    print()

    # --------------------------------------------------------
    # Transfer penalty
    # --------------------------------------------------------

    fixed_dcf_penalty = (
        test_at_dcf["normalized_DCF"]
        - test_oracle_dcf["normalized_DCF"]
    )

    fixed_balanced_error_penalty = (
        test_at_eer["balanced_error_percent"]
        - test_oracle_eer["EER_percent"]
    )

    print("=" * 96)
    print("TRANSFER PENALTY")
    print("=" * 96)

    print(
        "EER-oriented fixed-threshold "
        "balanced-error penalty vs test oracle: "
        f"{fixed_balanced_error_penalty:+.6f} pp"
    )

    print(
        "DCF-oriented fixed-threshold "
        "normalized-DCF penalty vs test oracle: "
        f"{fixed_dcf_penalty:+.9f}"
    )

    print()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    out = root / CFG.output_rel
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_path = (
        out
        / "r2b_fixed_threshold_results.csv"
    )

    summary_path = (
        out
        / "r2b_summary.json"
    )

    policy_path = (
        out
        / "r2b_cal_frozen_decision_thresholds.json"
    )

    result_df.to_csv(
        result_path,
        index=False,
    )

    policy = {
        "routing_policy":
            "Original-Frozen-B10",

        "routing_tau_b0":
            CFG.tau_b0,

        "routing_delta_b10":
            CFG.delta_b10,

        "final_decision_thresholds_from_CAL": {
            "EER_oriented":
                theta_eer,
            "minDCF_oriented":
                theta_dcf,
        },

        "important":
            "Routing thresholds and final decision "
            "thresholds are distinct.",
    }

    policy_path.write_text(
        json.dumps(
            policy,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = {
        "stage":
            "R2-B",

        "status":
            "COMPLETE_NOT_YET_LOCKED",

        "protocol": {
            "routing_modified":
                False,

            "decision_threshold_source":
                "CAL",

            "test_threshold_retuning":
                False,

            "platt_calibration":
                False,
        },

        "cal_frozen_thresholds": {
            "theta_mixed_eer":
                theta_eer,

            "theta_mixed_dcf":
                theta_dcf,
        },

        "cal_results": {
            "EER_threshold":
                cal_at_eer,

            "minDCF_threshold":
                cal_at_dcf,
        },

        "vox1o_fixed_threshold_results": {
            "EER_threshold":
                test_at_eer,

            "minDCF_threshold":
                test_at_dcf,
        },

        "vox1o_oracle_reference": {
            "EER":
                test_oracle_eer,

            "minDCF":
                test_oracle_dcf,
        },

        "transfer_penalty": {
            "balanced_error_pp":
                fixed_balanced_error_penalty,

            "normalized_DCF":
                fixed_dcf_penalty,
        },
    }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 96)
    print("ARTIFACTS")
    print("=" * 96)

    print(result_path)
    print(policy_path)
    print(summary_path)

    print()

    print("=" * 96)
    print(
        "STAGE R2-B COMPLETE — NOT YET LOCKED"
    )
    print("=" * 96)

    print(
        "Do NOT change the decision thresholds "
        "after reviewing VoxCeleb1-O."
    )


if __name__ == "__main__":
    main()