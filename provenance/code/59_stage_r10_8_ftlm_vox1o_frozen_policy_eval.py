from pathlib import Path
import json
import hashlib

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"package/")

SCORES = (
    ROOT
    / "results"
    / "stage_r10_ftlm"
    / "vox1o_scores"
    / "vox1o_ftlm_b0_b2_b6_scores.csv"
)

POLICIES = (
    ROOT
    / "results"
    / "stage_r10_ftlm"
    / "policy_freeze"
    / "ftlm_frozen_routing_policies.csv"
)

OUT_DIR = (
    ROOT
    / "results"
    / "stage_r10_ftlm"
    / "vox1o_frozen_policy_eval"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "vox1o_ftlm_frozen_policy_results.csv"
OUT_TRIALS = OUT_DIR / "vox1o_ftlm_frozen_policy_trial_audit.csv"
OUT_JSON = OUT_DIR / "r10_8_report.json"


# ============================================================
# METRICS
# ============================================================

P_TARGET = 0.01
C_MISS = 1.0
C_FA = 1.0


def eer_metric(labels, scores):
    fpr, tpr, thresholds = roc_curve(
        labels,
        scores,
        pos_label=1,
    )

    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))

    eer = (fpr[idx] + fnr[idx]) / 2.0

    return float(eer * 100.0), float(thresholds[idx])


def mindcf_metric(labels, scores):
    fpr, tpr, thresholds = roc_curve(
        labels,
        scores,
        pos_label=1,
    )

    fnr = 1.0 - tpr

    costs = (
        C_MISS * P_TARGET * fnr
        + C_FA * (1.0 - P_TARGET) * fpr
    )

    # Normalized DCF.
    normalizer = min(
        C_MISS * P_TARGET,
        C_FA * (1.0 - P_TARGET),
    )

    dcf = costs / normalizer

    idx = np.argmin(dcf)

    return float(dcf[idx]), float(thresholds[idx])


def evaluate(labels, scores):
    eer, eer_thr = eer_metric(labels, scores)
    dcf, dcf_thr = mindcf_metric(labels, scores)

    return {
        "EER_percent": eer,
        "minDCF": dcf,
        "AUC": float(roc_auc_score(labels, scores)),
        "EER_threshold": eer_thr,
        "minDCF_threshold": dcf_thr,
    }


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)

    return h.hexdigest()


def find_column(df, candidates, description):
    lower_map = {str(c).lower(): c for c in df.columns}

    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    raise KeyError(
        f"Could not identify {description}. "
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 96)
    print("R10-8 — VOXCELEB1-O FT-LM FROZEN POLICY EVALUATION")
    print("=" * 96)

    print("SCORES   :", SCORES)
    print("POLICIES :", POLICIES)

    if not SCORES.exists():
        raise FileNotFoundError(SCORES)

    if not POLICIES.exists():
        raise FileNotFoundError(POLICIES)

    score_sha = sha256_file(SCORES)
    policy_sha = sha256_file(POLICIES)

    print("\nSHA256")
    print("SCORES   :", score_sha)
    print("POLICIES :", policy_sha)

    # --------------------------------------------------------
    # Load score table
    # --------------------------------------------------------

    df = pd.read_csv(SCORES)

    required = {
        "trial_id",
        "label",
        "score_b0",
        "score_b6",
    }

    missing = required - set(df.columns)

    if missing:
        raise KeyError(
            f"Missing score columns: {sorted(missing)}"
        )

    if len(df) != 37720:
        raise RuntimeError(
            f"Expected 37720 VoxCeleb1-O trials, found {len(df)}"
        )

    if df["trial_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate trial_id detected."
        )

    labels = df["label"].to_numpy(dtype=int)
    s0 = df["score_b0"].to_numpy(dtype=float)
    s6 = df["score_b6"].to_numpy(dtype=float)

    if not np.isfinite(s0).all():
        raise RuntimeError("Non-finite b0 scores detected.")

    if not np.isfinite(s6).all():
        raise RuntimeError("Non-finite b6 scores detected.")

    n_target = int(np.sum(labels == 1))
    n_impostor = int(np.sum(labels == 0))

    print("\nDATA")
    print("TRIALS   :", len(df))
    print("TARGET   :", n_target)
    print("IMPOSTOR :", n_impostor)

    # --------------------------------------------------------
    # Static references
    # --------------------------------------------------------

    static_b0 = evaluate(labels, s0)
    static_b6 = evaluate(labels, s6)

    print("\n" + "=" * 96)
    print("STATIC FT-LM REFERENCES")
    print("=" * 96)

    print(
        f"b0 | "
        f"EER={static_b0['EER_percent']:.4f}% | "
        f"minDCF={static_b0['minDCF']:.6f} | "
        f"AUC={static_b0['AUC']:.6f}"
    )

    print(
        f"b6 | "
        f"EER={static_b6['EER_percent']:.4f}% | "
        f"minDCF={static_b6['minDCF']:.6f} | "
        f"AUC={static_b6['AUC']:.6f}"
    )

    # --------------------------------------------------------
    # Load frozen policies
    # --------------------------------------------------------

    policies = pd.read_csv(POLICIES)

    policy_col = find_column(
        policies,
        [
            "policy",
            "policy_name",
            "name",
        ],
        "policy-name column",
    )

    delta_col = find_column(
        policies,
        [
            "delta",
            "delta_dev",
            "routing_delta",
            "margin_cutoff",
            "frozen_delta",
        ],
        "frozen routing-delta column",
    )

    # Decision threshold column may be stored explicitly.
    # If not, use the already frozen values from R10-7E.
    tau_col = None

    for candidate in [
        "tau_b0",
        "b0_threshold",
        "decision_threshold_b0",
        "frozen_b0_threshold",
    ]:
        for col in policies.columns:
            if col.lower() == candidate.lower():
                tau_col = col
                break
        if tau_col is not None:
            break

    # R10-7E LOCK values.
    FROZEN_TAU = {
        "EER": 0.284263853,
        "DCF": 0.435542784,
    }

    print("\n" + "=" * 96)
    print("FROZEN POLICY EVALUATION")
    print("=" * 96)

    rows = []

    audit = df[
        [
            "trial_id",
            "label",
            "score_b0",
            "score_b6",
        ]
    ].copy()

    for _, p in policies.iterrows():

        policy_name = str(p[policy_col])
        delta = float(p[delta_col])

        upper_name = policy_name.upper()

        if upper_name.startswith("EER"):
            regime = "EER"
        elif upper_name.startswith("DCF"):
            regime = "DCF"
        else:
            raise RuntimeError(
                f"Unknown policy regime: {policy_name}"
            )

        if tau_col is not None and pd.notna(p[tau_col]):
            tau_b0 = float(p[tau_col])
        else:
            tau_b0 = FROZEN_TAU[regime]

        # ----------------------------------------------------
        # Frozen routing rule
        # ----------------------------------------------------

        margin = np.abs(s0 - tau_b0)

        route = margin <= delta

        n_route = int(route.sum())
        usage = 100.0 * n_route / len(df)

        mixed = s0.copy()
        mixed[route] = s6[route]

        metrics = evaluate(labels, mixed)

        # ----------------------------------------------------
        # Relative EER recovery
        # ----------------------------------------------------

        denom = (
            static_b0["EER_percent"]
            - static_b6["EER_percent"]
        )

        if abs(denom) > 1e-12:
            eer_recovery = (
                100.0
                * (
                    static_b0["EER_percent"]
                    - metrics["EER_percent"]
                )
                / denom
            )
        else:
            eer_recovery = np.nan

        # ----------------------------------------------------
        # Relative minDCF recovery
        # ----------------------------------------------------

        dcf_denom = (
            static_b0["minDCF"]
            - static_b6["minDCF"]
        )

        if abs(dcf_denom) > 1e-12:
            dcf_recovery = (
                100.0
                * (
                    static_b0["minDCF"]
                    - metrics["minDCF"]
                )
                / dcf_denom
            )
        else:
            dcf_recovery = np.nan

        row = {
            "policy": policy_name,
            "regime": regime,
            "frozen_tau_b0": tau_b0,
            "frozen_delta": delta,
            "routed_trials": n_route,
            "observed_b6_usage_percent": usage,
            "EER_percent": metrics["EER_percent"],
            "minDCF": metrics["minDCF"],
            "AUC": metrics["AUC"],
            "EER_threshold_test_sweep": metrics[
                "EER_threshold"
            ],
            "minDCF_threshold_test_sweep": metrics[
                "minDCF_threshold"
            ],
            "EER_gap_recovery_percent": eer_recovery,
            "minDCF_gap_recovery_percent": dcf_recovery,
        }

        rows.append(row)

        audit[f"{policy_name}_margin"] = margin
        audit[f"{policy_name}_route"] = route.astype(np.int8)
        audit[f"{policy_name}_mixed_score"] = mixed

        print(
            f"{policy_name:8s} | "
            f"delta={delta:.9f} | "
            f"b6={usage:7.3f}% ({n_route:5d}) | "
            f"EER={metrics['EER_percent']:.4f}% | "
            f"minDCF={metrics['minDCF']:.6f} | "
            f"AUC={metrics['AUC']:.6f} | "
            f"EER-rec={eer_recovery:7.2f}% | "
            f"DCF-rec={dcf_recovery:7.2f}%"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    result_df = pd.DataFrame(rows)

    result_df.to_csv(
        OUT_CSV,
        index=False,
    )

    audit.to_csv(
        OUT_TRIALS,
        index=False,
    )

    report = {
        "stage": "R10-8",
        "protocol": "VoxCeleb1-O",
        "trials": int(len(df)),
        "target": n_target,
        "impostor": n_impostor,
        "score_file": str(SCORES),
        "score_sha256": score_sha,
        "policy_file": str(POLICIES),
        "policy_sha256": policy_sha,
        "static_b0": static_b0,
        "static_b6": static_b6,
        "policies": rows,
        "firewall": (
            "All routing cutoffs were frozen before "
            "VoxCeleb1-O evaluation. No test-set "
            "retuning was performed."
        ),
    }

    with open(
        OUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    print("\n" + "=" * 96)
    print("ARTIFACTS")
    print("=" * 96)

    print("RESULTS :", OUT_CSV)
    print("TRIALS  :", OUT_TRIALS)
    print("REPORT  :", OUT_JSON)

    print("\n=== R10-8 PASS / TEST FIREWALL PRESERVED ===")


if __name__ == "__main__":
    main()