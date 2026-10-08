import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


# ============================================================
# PROJECT / INPUT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CAL_SCORES = (
    PROJECT_ROOT / "results" / "stage03_dev_cal"
    / "stage03f_scores" / "cal_b0_b6_scores_20k.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "stage03_dev_cal"
    / "stage03g_thresholds"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUTPUT_DIR / "cal_frozen_thresholds.csv"
OUT_JSON = OUTPUT_DIR / "cal_frozen_thresholds.json"


# ============================================================
# POLICY
# ============================================================

# Primary frozen decision threshold:
# EER operating point estimated ONLY on CAL.
#
# minDCF threshold is also recorded as a secondary operating point,
# but it is NOT silently substituted for the primary threshold.
P_TARGET = 0.01
C_MISS = 1.0
C_FA = 1.0


def eer_threshold(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return {
        "eer_percent": float(100.0 * (fpr[idx] + fnr[idx]) / 2.0),
        "threshold": float(thresholds[idx]),
        "far": float(fpr[idx]),
        "frr": float(fnr[idx]),
    }


def mindcf_threshold(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    cost = C_MISS * fnr * P_TARGET + C_FA * fpr * (1.0 - P_TARGET)
    default = min(C_MISS * P_TARGET, C_FA * (1.0 - P_TARGET))
    idx = int(np.argmin(cost))
    return {
        "minDCF": float(cost[idx] / default),
        "threshold": float(thresholds[idx]),
        "far": float(fpr[idx]),
        "frr": float(fnr[idx]),
    }


def main():
    print("=" * 92)
    print("STAGE 03G-1 — CALIBRATION-ONLY DECISION THRESHOLD FREEZING")
    print("=" * 92)

    if not CAL_SCORES.exists():
        raise FileNotFoundError(CAL_SCORES)

    df = pd.read_csv(CAL_SCORES)
    required = {"trial_id", "label", "score_b0", "score_b6"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns: {sorted(missing)}")

    if len(df) != 20000:
        raise RuntimeError(f"Expected 20000 CAL trials, found {len(df)}")

    if df["trial_id"].duplicated().any():
        raise RuntimeError("Duplicate CAL trial_id detected.")

    x = df[["label", "score_b0", "score_b6"]].to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise RuntimeError("NaN/non-finite CAL values detected.")

    labels = df["label"].to_numpy(dtype=int)
    if set(np.unique(labels)) != {0, 1}:
        raise RuntimeError("CAL labels must contain both 0 and 1.")

    rows = []
    report = {
        "stage": "03G-1",
        "source_split": "CAL",
        "source_trials": int(len(df)),
        "primary_policy": "CAL-derived EER threshold",
        "secondary_policy": "CAL-derived minDCF threshold",
        "minDCF_config": {
            "P_target": P_TARGET,
            "C_miss": C_MISS,
            "C_fa": C_FA,
        },
        "models": {},
    }

    for model, col in [("b0", "score_b0"), ("b6", "score_b6")]:
        scores = df[col].to_numpy(dtype=float)
        eer = eer_threshold(labels, scores)
        dcf = mindcf_threshold(labels, scores)

        rows.append({
            "model": model,
            "primary_threshold_type": "EER",
            "frozen_EER_threshold": eer["threshold"],
            "CAL_EER_percent": eer["eer_percent"],
            "CAL_EER_FAR": eer["far"],
            "CAL_EER_FRR": eer["frr"],
            "secondary_minDCF_threshold": dcf["threshold"],
            "CAL_minDCF": dcf["minDCF"],
            "CAL_minDCF_FAR": dcf["far"],
            "CAL_minDCF_FRR": dcf["frr"],
        })

        report["models"][model] = {
            "frozen_primary_EER": eer,
            "secondary_minDCF": dcf,
        }

        print(
            f"ReDimNet-{model} | "
            f"FROZEN EER threshold={eer['threshold']:.8f} | "
            f"CAL EER={eer['eer_percent']:.4f}% | "
            f"secondary minDCF threshold={dcf['threshold']:.8f} | "
            f"minDCF={dcf['minDCF']:.6f}"
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    report["lock"] = (
        "Primary b0/b6 verification decision thresholds are frozen from CAL only. "
        "They may be applied to DEV for benefit/complementarity analysis and to "
        "VoxCeleb1-O for final fixed-policy evaluation, but must not be retuned "
        "using DEV or VoxCeleb1-O outcomes."
    )

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print()
    print("Primary frozen thresholds:")
    for row in rows:
        print(f"{row['model']}: {row['frozen_EER_threshold']:.8f}")

    print()
    print(f"CSV : {OUT_CSV}")
    print(f"JSON: {OUT_JSON}")
    print()
    print("PASS — CAL-only b0/b6 decision thresholds frozen.")
    print("=" * 92)


if __name__ == "__main__":
    main()
