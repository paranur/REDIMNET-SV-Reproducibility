import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_SCORES = (
    PROJECT_ROOT / "results" / "stage01_clean_vox1o"
    / "scores" / "vox1o_b0_b6_scores.csv"
)

FROZEN_THRESHOLDS = (
    PROJECT_ROOT / "results" / "stage03_dev_cal"
    / "stage03g_thresholds" / "cal_frozen_thresholds.csv"
)

FROZEN_POLICIES = (
    PROJECT_ROOT / "results" / "stage03_dev_cal"
    / "stage03h_policy_freeze" / "dev_margin_frozen_policies.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "stage04_final_test" / "stage04a_frozen_router"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUTPUT_DIR / "vox1o_frozen_margin_router_results.csv"
OUT_JSON = OUTPUT_DIR / "vox1o_frozen_margin_router_results.json"
OUT_TRIALS = OUTPUT_DIR / "vox1o_frozen_margin_router_trial_audit.csv"

P_TARGET = 0.01
C_MISS = 1.0
C_FA = 1.0


def eer_metric(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return 100.0 * float((fpr[idx] + fnr[idx]) / 2.0), float(thresholds[idx])


def mindcf_metric(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    costs = C_MISS * fnr * P_TARGET + C_FA * fpr * (1.0 - P_TARGET)
    default = min(C_MISS * P_TARGET, C_FA * (1.0 - P_TARGET))
    idx = int(np.argmin(costs))
    return float(costs[idx] / default), float(thresholds[idx])


def evaluate(labels, scores):
    eer, eer_thr = eer_metric(labels, scores)
    dcf, dcf_thr = mindcf_metric(labels, scores)
    return {
        "EER_percent": eer,
        "EER_threshold": eer_thr,
        "minDCF": dcf,
        "minDCF_threshold": dcf_thr,
        "AUC": float(roc_auc_score(labels, scores)),
    }


def get_tau_b0(threshold_df):
    # Robustly support likely column layouts.
    cols = set(threshold_df.columns)

    if {"model", "frozen_EER_threshold"}.issubset(cols):
        row = threshold_df[
            threshold_df["model"].astype(str).str.lower().isin(
                ["redimnet-b0", "redimnet_b0", "b0"]
            )
        ]

        if len(row) != 1:
            raise RuntimeError("Could not uniquely identify b0 threshold row.")

        return float(row.iloc[0]["frozen_EER_threshold"])

    if {"model", "EER_threshold"}.issubset(cols):
        row = threshold_df[threshold_df["model"].astype(str).str.lower().isin(
            ["redimnet-b0", "redimnet_b0", "b0"]
        )]
        if len(row) != 1:
            raise RuntimeError("Could not uniquely identify b0 threshold row.")
        return float(row.iloc[0]["EER_threshold"])

    # Known frozen value is NOT hard-coded deliberately.
    raise RuntimeError(
        f"Unsupported threshold CSV schema: {list(threshold_df.columns)}"
    )


def main():
    print("=" * 105)
    print("STAGE 04A — UNTOUCHED VOXCELEB1-O EVALUATION WITH FROZEN DEV ROUTING POLICIES")
    print("=" * 105)

    for p in [TEST_SCORES, FROZEN_THRESHOLDS, FROZEN_POLICIES]:
        if not p.exists():
            raise FileNotFoundError(p)

    test = pd.read_csv(TEST_SCORES)
    th = pd.read_csv(FROZEN_THRESHOLDS)
    policies = pd.read_csv(FROZEN_POLICIES)

    required_test = {"trial_id", "label", "score_b0", "score_b6"}
    missing = required_test - set(test.columns)
    if missing:
        raise RuntimeError(f"Missing test columns: {sorted(missing)}")

    required_pol = {"policy_name", "delta_margin_cutoff"}
    missing = required_pol - set(policies.columns)
    if missing:
        raise RuntimeError(f"Missing policy columns: {sorted(missing)}")

    if len(test) != 37720:
        raise RuntimeError(f"Expected 37720 VoxCeleb1-O trials, found {len(test)}")
    if test["trial_id"].duplicated().any():
        raise RuntimeError("Duplicate VoxCeleb1-O trial_id detected.")

    labels = test["label"].to_numpy(dtype=int)
    s0 = test["score_b0"].to_numpy(dtype=float)
    s6 = test["score_b6"].to_numpy(dtype=float)

    if not np.isfinite(np.column_stack([labels, s0, s6])).all():
        raise RuntimeError("NaN/non-finite values detected in test scores.")

    tau_b0 = get_tau_b0(th)
    margin = np.abs(s0 - tau_b0)

    static_b0 = evaluate(labels, s0)
    static_b6 = evaluate(labels, s6)

    print(f"Frozen CAL tau_b0: {tau_b0:.9f}")
    print()
    print(
        f"Static b0 | EER={static_b0['EER_percent']:.4f}% | "
        f"minDCF={static_b0['minDCF']:.6f} | AUC={static_b0['AUC']:.6f}"
    )
    print(
        f"Static b6 | EER={static_b6['EER_percent']:.4f}% | "
        f"minDCF={static_b6['minDCF']:.6f} | AUC={static_b6['AUC']:.6f}"
    )
    print()

    rows = []
    trial_audit = test[["trial_id", "label", "score_b0", "score_b6"]].copy()
    trial_audit["b0_abs_margin_frozen_tau"] = margin

    denom = static_b0["EER_percent"] - static_b6["EER_percent"]

    for _, pol in policies.iterrows():
        name = str(pol["policy_name"])
        delta = float(pol["delta_margin_cutoff"])

        # CRITICAL: no test ranking, no test quota.
        route = margin <= delta
        n_route = int(route.sum())
        usage = 100.0 * n_route / len(test)

        adaptive = s0.copy()
        adaptive[route] = s6[route]

        metrics = evaluate(labels, adaptive)

        recovery = (
            100.0 * (static_b0["EER_percent"] - metrics["EER_percent"]) / denom
            if denom != 0 else np.nan
        )

        rows.append({
            "policy_name": name,
            "frozen_delta": delta,
            "observed_b6_usage_percent": usage,
            "observed_b6_routed_trials": n_route,
            **metrics,
            "test_EER_gap_recovery_percent": recovery,
        })

        trial_audit[f"route_{name}"] = route.astype(int)

        print(
            f"{name:>12} | delta={delta:.9f} | "
            f"observed b6={usage:7.3f}% ({n_route:5d}) | "
            f"EER={metrics['EER_percent']:.4f}% | "
            f"minDCF={metrics['minDCF']:.6f} | "
            f"AUC={metrics['AUC']:.6f} | "
            f"EER-gap recovery={recovery:7.2f}%"
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    trial_audit.to_csv(OUT_TRIALS, index=False)

    report = {
        "stage": "04A",
        "dataset": "VoxCeleb1-O",
        "trials": int(len(test)),
        "protocol": "untouched final test; no test ranking or retuning",
        "frozen_CAL_tau_b0": tau_b0,
        "static_b0": static_b0,
        "static_b6": static_b6,
        "policies": rows,
        "critical_rule": (
            "Each test trial is independently routed using the DEV-frozen "
            "absolute-margin cutoff. Observed b6 usage is not forced."
        ),
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print()
    print(f"Summary    : {OUT_CSV}")
    print(f"Trial audit: {OUT_TRIALS}")
    print(f"JSON       : {OUT_JSON}")
    print()
    print("PASS — Stage 04A frozen-policy VoxCeleb1-O evaluation completed.")
    print("=" * 105)


if __name__ == "__main__":
    main()
