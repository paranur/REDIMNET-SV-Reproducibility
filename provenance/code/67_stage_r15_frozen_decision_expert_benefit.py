from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


# =============================================================================
# R15 — HELD-OUT FROZEN-DECISION EXPERT-BENEFIT VALIDATION
#
# Purpose
# -------
# Reviewer-response experiment:
#
#   1. Evaluate expert benefit on held-out VoxCeleb1-O.
#   2. Use the actual frozen B10 deployed decision rule.
#   3. Separate beneficial/harmful interventions into FA and MISS cases.
#   4. Validate margin -> deployed-benefit ranking.
#   5. Report benefit/harm capture by the frozen B10 router.
#   6. Add speaker-aware bootstrap uncertainty.
#
# IMPORTANT
# ---------
# No threshold is estimated from VoxCeleb1-O.
# No routing cutoff is estimated from VoxCeleb1-O.
# VoxCeleb1-O is evaluation only.
# =============================================================================


@dataclass(frozen=True)
class Config:
    # -------------------------------------------------------------------------
    # Project
    # -------------------------------------------------------------------------
    PROJECT_ROOT: Path = Path(r"package/")

    # -------------------------------------------------------------------------
    # Canonical frozen-router audit
    # -------------------------------------------------------------------------
    TRIAL_AUDIT_REL: Path = Path(
        r"results\stage04_final_test\stage04a_frozen_router"
        r"\vox1o_frozen_margin_router_trial_audit.csv"
    )

    # -------------------------------------------------------------------------
    # Canonical VoxCeleb1-O protocol
    # -------------------------------------------------------------------------
    VOX1O_PROTOCOL: Path = Path(
        r"voxceleb1/veri_test.txt"
    )

    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------
    OUT_REL: Path = Path(
        r"results\reviewer_audit\r15_frozen_decision_expert_benefit"
    )

    # -------------------------------------------------------------------------
    # CAL-frozen decision thresholds
    # -------------------------------------------------------------------------
    TAU_B0: float = 0.336179673672
    TAU_B6: float = 0.340364754200

    # Final mixed-system threshold estimated on CAL.
    TAU_B10_MIXED: float = 0.338037937880

    # DEV-frozen B10 routing cutoff
    DELTA_B10: float = 0.134631842

    # -------------------------------------------------------------------------
    # Bootstrap
    # -------------------------------------------------------------------------
    N_BOOT: int = 2000
    SEED: int = 20260921


# =============================================================================
# Utilities
# =============================================================================


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_label(x) -> int:
    """
    Normalize common verification label encodings to:
        1 = target
        0 = impostor
    """
    if isinstance(x, str):
        s = x.strip().lower()

        if s in {"1", "target", "true", "same", "positive"}:
            return 1

        if s in {"0", "impostor", "nontarget", "non-target",
                 "false", "different", "negative"}:
            return 0

    v = int(x)

    if v not in (0, 1):
        raise ValueError(f"Unsupported label: {x}")

    return v


def speaker_from_path(path: str) -> str:
    """
    VoxCeleb path is normally:
        id10270/xxxx/00001.wav

    Speaker ID = first path component.
    """
    p = str(path).strip().replace("\\", "/")
    parts = [x for x in p.split("/") if x]

    if not parts:
        raise ValueError(f"Cannot extract speaker from path: {path}")

    return parts[0]


def read_vox1o_protocol(path: Path) -> pd.DataFrame:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 3:
                raise RuntimeError(
                    f"Malformed protocol line {line_no}: {line}"
                )

            label = normalize_label(parts[0])
            enroll_path = parts[1]
            test_path = parts[2]

            enroll_spk = speaker_from_path(enroll_path)
            test_spk = speaker_from_path(test_path)

            # Independent sanity check.
            expected_label = int(enroll_spk == test_spk)

            if label != expected_label:
                raise RuntimeError(
                    f"Protocol speaker/label mismatch at line {line_no}: "
                    f"label={label}, enroll={enroll_spk}, test={test_spk}"
                )

            rows.append(
                {
                    "protocol_index": len(rows),
                    "protocol_label": label,
                    "enroll_path": enroll_path,
                    "test_path": test_path,
                    "enroll_spk": enroll_spk,
                    "test_spk": test_spk,
                }
            )

    return pd.DataFrame(rows)


def decision(score: np.ndarray, threshold: float) -> np.ndarray:
    return (score >= threshold).astype(np.int8)


def correctness(pred: np.ndarray, label: np.ndarray) -> np.ndarray:
    return pred == label


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return float("nan"), float("nan")

    return (
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    )


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y)

    if len(np.unique(y)) < 2:
        return float("nan")

    return float(roc_auc_score(y, score))


def safe_ap(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y)

    if len(np.unique(y)) < 2:
        return float("nan")

    return float(average_precision_score(y, score))


# =============================================================================
# Metric computation
# =============================================================================


def compute_metrics(df: pd.DataFrame) -> dict:
    label = df["label"].to_numpy(dtype=np.int8)

    b0_pred = df["b0_pred"].to_numpy(dtype=np.int8)
    mixed_pred = df["mixed_pred"].to_numpy(dtype=np.int8)

    routed = df["route_b10"].to_numpy(dtype=bool)

    benefit = df["benefit"].to_numpy(dtype=np.int8)
    harm = df["harm"].to_numpy(dtype=np.int8)

    # Smaller margin = more likely benefit.
    ranking_score = -df["b0_margin"].to_numpy(dtype=float)

    n = len(df)

    b0_correct = b0_pred == label
    mixed_correct = mixed_pred == label

    # -------------------------------------------------------------------------
    # Decision transition table:
    #
    # CC: b0 correct -> mixed correct
    # CW: b0 correct -> mixed wrong  = harm
    # WC: b0 wrong   -> mixed correct = benefit
    # WW: b0 wrong   -> mixed wrong
    # -------------------------------------------------------------------------
    cc = int(np.sum(b0_correct & mixed_correct))
    cw = int(np.sum(b0_correct & ~mixed_correct))
    wc = int(np.sum(~b0_correct & mixed_correct))
    ww = int(np.sum(~b0_correct & ~mixed_correct))

    # -------------------------------------------------------------------------
    # Benefit/harm decomposition
    #
    # label=0:
    #   pred=1 -> false accept
    #
    # label=1:
    #   pred=0 -> miss / false reject
    # -------------------------------------------------------------------------

    benefit_fa_corrected = int(
        np.sum(
            (label == 0)
            & (b0_pred == 1)
            & (mixed_pred == 0)
        )
    )

    benefit_miss_corrected = int(
        np.sum(
            (label == 1)
            & (b0_pred == 0)
            & (mixed_pred == 1)
        )
    )

    harm_fa_created = int(
        np.sum(
            (label == 0)
            & (b0_pred == 0)
            & (mixed_pred == 1)
        )
    )

    harm_miss_created = int(
        np.sum(
            (label == 1)
            & (b0_pred == 1)
            & (mixed_pred == 0)
        )
    )

    n_benefit = int(np.sum(benefit))
    n_harm = int(np.sum(harm))

    routed_n = int(np.sum(routed))

    benefit_routed = int(np.sum((benefit == 1) & routed))
    harm_routed = int(np.sum((harm == 1) & routed))

    benefit_capture = (
        benefit_routed / n_benefit if n_benefit else np.nan
    )

    harm_capture = (
        harm_routed / n_harm if n_harm else np.nan
    )

    benefit_precision = (
        benefit_routed / routed_n if routed_n else np.nan
    )

    net_improvement = n_benefit - n_harm

    return {
        "n_trials": int(n),

        "n_target": int(np.sum(label == 1)),
        "n_impostor": int(np.sum(label == 0)),

        "routed_n": routed_n,
        "routed_pct": 100.0 * routed_n / n,

        "CC": cc,
        "CW": cw,
        "WC": wc,
        "WW": ww,

        "benefit_n": n_benefit,
        "benefit_pct": 100.0 * n_benefit / n,

        "harm_n": n_harm,
        "harm_pct": 100.0 * n_harm / n,

        "net_corrected_decisions": net_improvement,
        "net_corrected_pct": 100.0 * net_improvement / n,

        "benefit_FA_corrected": benefit_fa_corrected,
        "benefit_MISS_corrected": benefit_miss_corrected,

        "harm_FA_created": harm_fa_created,
        "harm_MISS_created": harm_miss_created,

        "benefit_roc_auc": safe_auc(benefit, ranking_score),
        "benefit_pr_auc": safe_ap(benefit, ranking_score),

        "benefit_routed_n": benefit_routed,
        "benefit_capture": float(benefit_capture),
        "benefit_capture_pct": float(100.0 * benefit_capture),

        "harm_routed_n": harm_routed,
        "harm_capture": float(harm_capture),
        "harm_capture_pct": float(100.0 * harm_capture),

        "benefit_precision": float(benefit_precision),
        "benefit_precision_pct": float(100.0 * benefit_precision),

        "b0_accuracy_pct": float(100.0 * np.mean(b0_correct)),
        "mixed_accuracy_pct": float(100.0 * np.mean(mixed_correct)),
    }


# =============================================================================
# Speaker-aware bootstrap
# =============================================================================


def bootstrap_speaker_aware(
    df: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    """
    Alternative speaker-aware bootstrap consistent with the R13 sensitivity
    principle:

    - sample VoxCeleb1-O speakers with replacement;
    - a trial is retained if at least one endpoint belongs to a sampled
      speaker;
    - retained trials receive unit weight;
    - no m_i * m_j dyadic multiplicity weighting.

    This is used here for uncertainty sensitivity rather than to create or
    retune any decision/routing parameter.
    """

    speakers = sorted(
        set(df["enroll_spk"].astype(str))
        | set(df["test_spk"].astype(str))
    )

    speakers = np.asarray(speakers, dtype=object)

    enroll = df["enroll_spk"].astype(str).to_numpy()
    test = df["test_spk"].astype(str).to_numpy()

    rng = np.random.default_rng(seed)

    records = []

    for b in range(n_boot):
        sampled = rng.choice(
            speakers,
            size=len(speakers),
            replace=True,
        )

        sampled_unique = set(sampled.tolist())

        mask = np.fromiter(
            (
                (e in sampled_unique) or (t in sampled_unique)
                for e, t in zip(enroll, test)
            ),
            dtype=bool,
            count=len(df),
        )

        sub = df.loc[mask]

        if len(sub) == 0:
            continue

        m = compute_metrics(sub)

        records.append(
            {
                "bootstrap": b,
                "n_trials": m["n_trials"],
                "routed_pct": m["routed_pct"],
                "benefit_n": m["benefit_n"],
                "harm_n": m["harm_n"],
                "net_corrected_decisions": m["net_corrected_decisions"],
                "benefit_roc_auc": m["benefit_roc_auc"],
                "benefit_pr_auc": m["benefit_pr_auc"],
                "benefit_capture_pct": m["benefit_capture_pct"],
                "harm_capture_pct": m["harm_capture_pct"],
                "benefit_precision_pct": m["benefit_precision_pct"],
            }
        )

    return pd.DataFrame(records)


def summarize_bootstrap(boot: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "routed_pct",
        "benefit_n",
        "harm_n",
        "net_corrected_decisions",
        "benefit_roc_auc",
        "benefit_pr_auc",
        "benefit_capture_pct",
        "harm_capture_pct",
        "benefit_precision_pct",
    ]

    rows = []

    for metric in metrics:
        x = boot[metric].to_numpy(dtype=float)
        x = x[np.isfinite(x)]

        lo, hi = percentile_ci(x)

        rows.append(
            {
                "metric": metric,
                "bootstrap_mean": float(np.mean(x)),
                "bootstrap_std": float(np.std(x, ddof=1)),
                "ci95_low": lo,
                "ci95_high": hi,
                "n_valid": int(len(x)),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    cfg = Config()

    audit_path = cfg.PROJECT_ROOT / cfg.TRIAL_AUDIT_REL
    out_dir = cfg.PROJECT_ROOT / cfg.OUT_REL

    ensure_dir(out_dir)

    print("=" * 100)
    print("R15 — HELD-OUT FROZEN-DECISION EXPERT-BENEFIT VALIDATION")
    print("=" * 100)

    print("\nINPUTS")
    print("-" * 100)
    print("Trial audit :", audit_path)
    print("Protocol    :", cfg.VOX1O_PROTOCOL)
    print("Output      :", out_dir)

    if not audit_path.exists():
        raise FileNotFoundError(audit_path)

    if not cfg.VOX1O_PROTOCOL.exists():
        raise FileNotFoundError(cfg.VOX1O_PROTOCOL)

    # -------------------------------------------------------------------------
    # Load canonical frozen-router audit
    # -------------------------------------------------------------------------
    df = pd.read_csv(audit_path)

    required = {
        "trial_id",
        "label",
        "score_b0",
        "score_b6",
        "b0_abs_margin_frozen_tau",
        "route_MARGIN_B10",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing required trial-audit columns: {sorted(missing)}"
        )

    df["label"] = df["label"].map(normalize_label).astype(np.int8)

    print("\nTRIAL AUDIT")
    print("-" * 100)
    print("Rows        :", len(df))
    print("Targets     :", int((df["label"] == 1).sum()))
    print("Impostors   :", int((df["label"] == 0).sum()))

    if len(df) != 37720:
        raise RuntimeError(
            f"Expected 37,720 VoxCeleb1-O trials, found {len(df)}"
        )

    # -------------------------------------------------------------------------
    # Load protocol for speaker IDs
    # -------------------------------------------------------------------------
    protocol = read_vox1o_protocol(cfg.VOX1O_PROTOCOL)

    print("\nPROTOCOL AUDIT")
    print("-" * 100)
    print("Rows        :", len(protocol))
    print("Targets     :", int((protocol["protocol_label"] == 1).sum()))
    print("Impostors   :", int((protocol["protocol_label"] == 0).sum()))

    if len(protocol) != len(df):
        raise RuntimeError(
            "FAIL: protocol and frozen-router audit have different "
            f"trial counts: {len(protocol)} vs {len(df)}"
        )

    label_match = np.array_equal(
        df["label"].to_numpy(),
        protocol["protocol_label"].to_numpy(),
    )

    print("Label order :", "PASS" if label_match else "FAIL")

    if not label_match:
        bad = np.flatnonzero(
            df["label"].to_numpy()
            != protocol["protocol_label"].to_numpy()
        )

        print("First mismatched indices:", bad[:20].tolist())

        raise RuntimeError(
            "FAIL: veri_test.txt label order does not align with "
            "trial-audit CSV. Speaker metadata will NOT be attached."
        )

    # Attach protocol metadata only after alignment passes.
    for col in [
        "enroll_path",
        "test_path",
        "enroll_spk",
        "test_spk",
    ]:
        df[col] = protocol[col].to_numpy()

    all_speakers = sorted(
        set(df["enroll_spk"]) | set(df["test_spk"])
    )

    print("Unique speakers:", len(all_speakers))

    if len(all_speakers) != 40:
        print(
            "WARNING: canonical VoxCeleb1-O normally contains 40 speakers; "
            f"observed {len(all_speakers)}."
        )

    # -------------------------------------------------------------------------
    # Reconstruct frozen decisions
    # -------------------------------------------------------------------------
    score_b0 = df["score_b0"].to_numpy(dtype=float)
    score_b6 = df["score_b6"].to_numpy(dtype=float)

    df["b0_margin"] = np.abs(score_b0 - cfg.TAU_B0)

    stored_margin = df[
        "b0_abs_margin_frozen_tau"
    ].to_numpy(dtype=float)

    margin_error = np.max(
        np.abs(
            df["b0_margin"].to_numpy(dtype=float)
            - stored_margin
        )
    )

    print("\nFROZEN PARAMETER AUDIT")
    print("-" * 100)
    print(f"tau_b0      : {cfg.TAU_B0:.12f}")
    print(f"tau_b6      : {cfg.TAU_B6:.12f}")
    print(f"tau_B10_mix : {cfg.TAU_B10_MIXED:.12f}")
    print(f"delta_B10   : {cfg.DELTA_B10:.12f}")
    print(f"Max margin reconstruction error: {margin_error:.12e}")

    if margin_error > 1e-8:
        raise RuntimeError(
            "FAIL: reconstructed b0 margin does not match canonical "
            "trial-audit margin."
        )

    stored_route = (
        df["route_MARGIN_B10"].astype(int).to_numpy() == 1
    )

    reconstructed_route = (
        df["b0_margin"].to_numpy(dtype=float)
        <= cfg.DELTA_B10
    )

    route_match = np.array_equal(
        stored_route,
        reconstructed_route,
    )

    print("B10 route reconstruction:", "PASS" if route_match else "FAIL")

    if not route_match:
        n_diff = int(np.sum(stored_route != reconstructed_route))

        raise RuntimeError(
            f"FAIL: {n_diff} B10 routing decisions differ from the "
            "canonical trial-audit CSV."
        )

    df["route_b10"] = stored_route.astype(np.int8)

    # -------------------------------------------------------------------------
    # Actual deployed score:
    #
    # non-routed -> b0 score
    # routed     -> b6 score
    # -------------------------------------------------------------------------
    df["mixed_score"] = np.where(
        stored_route,
        score_b6,
        score_b0,
    )

    # -------------------------------------------------------------------------
    # Frozen decisions
    # -------------------------------------------------------------------------
    df["b0_pred"] = decision(
        score_b0,
        cfg.TAU_B0,
    )

    df["b6_pred"] = decision(
        score_b6,
        cfg.TAU_B6,
    )

    df["mixed_pred"] = decision(
        df["mixed_score"].to_numpy(dtype=float),
        cfg.TAU_B10_MIXED,
    )

    # -------------------------------------------------------------------------
    # Deployed-system benefit:
    #
    # benefit = b0 wrong, deployed mixed system correct
    # harm    = b0 correct, deployed mixed system wrong
    #
    # This is deliberately different from:
    #     b0 wrong & b6 correct using separate b0/b6 thresholds.
    # -------------------------------------------------------------------------
    label = df["label"].to_numpy(dtype=np.int8)

    b0_correct = (
        df["b0_pred"].to_numpy(dtype=np.int8) == label
    )

    mixed_correct = (
        df["mixed_pred"].to_numpy(dtype=np.int8) == label
    )

    df["benefit"] = (
        (~b0_correct) & mixed_correct
    ).astype(np.int8)

    df["harm"] = (
        b0_correct & (~mixed_correct)
    ).astype(np.int8)

    # -------------------------------------------------------------------------
    # Point estimates
    # -------------------------------------------------------------------------
    metrics = compute_metrics(df)

    print("\nDEPLOYED FROZEN-DECISION RESULTS")
    print("=" * 100)

    print(f"Trials                  : {metrics['n_trials']}")
    print(
        f"B10 routed              : {metrics['routed_n']} "
        f"({metrics['routed_pct']:.6f}%)"
    )

    print("\nDecision transitions")
    print("-" * 100)
    print(f"CC                       : {metrics['CC']}")
    print(f"CW / harmful             : {metrics['CW']}")
    print(f"WC / beneficial          : {metrics['WC']}")
    print(f"WW                       : {metrics['WW']}")

    print("\nBenefit / harm")
    print("-" * 100)
    print(
        f"Benefit                  : {metrics['benefit_n']} "
        f"({metrics['benefit_pct']:.6f}%)"
    )
    print(
        f"Harm                     : {metrics['harm_n']} "
        f"({metrics['harm_pct']:.6f}%)"
    )
    print(
        f"Net corrected decisions  : "
        f"{metrics['net_corrected_decisions']} "
        f"({metrics['net_corrected_pct']:.6f} pp)"
    )

    print("\nError-type decomposition")
    print("-" * 100)
    print(
        f"Benefit — FA corrected   : "
        f"{metrics['benefit_FA_corrected']}"
    )
    print(
        f"Benefit — MISS corrected : "
        f"{metrics['benefit_MISS_corrected']}"
    )
    print(
        f"Harm — FA created        : "
        f"{metrics['harm_FA_created']}"
    )
    print(
        f"Harm — MISS created      : "
        f"{metrics['harm_MISS_created']}"
    )

    print("\nBenefit ranking")
    print("-" * 100)
    print(
        f"Benefit ROC-AUC          : "
        f"{metrics['benefit_roc_auc']:.9f}"
    )
    print(
        f"Benefit PR-AUC           : "
        f"{metrics['benefit_pr_auc']:.9f}"
    )

    print("\nFrozen B10 capture")
    print("-" * 100)
    print(
        f"Benefit captured         : "
        f"{metrics['benefit_routed_n']} / "
        f"{metrics['benefit_n']} "
        f"= {metrics['benefit_capture_pct']:.6f}%"
    )
    print(
        f"Harm captured            : "
        f"{metrics['harm_routed_n']} / "
        f"{metrics['harm_n']} "
        f"= {metrics['harm_capture_pct']:.6f}%"
    )
    print(
        f"Benefit precision/routed : "
        f"{metrics['benefit_precision_pct']:.6f}%"
    )

    print("\nAccuracy")
    print("-" * 100)
    print(
        f"Frozen b0 accuracy       : "
        f"{metrics['b0_accuracy_pct']:.6f}%"
    )
    print(
        f"Frozen B10 accuracy      : "
        f"{metrics['mixed_accuracy_pct']:.6f}%"
    )

    # -------------------------------------------------------------------------
    # Speaker-aware bootstrap
    # -------------------------------------------------------------------------
    print("\nSPEAKER-AWARE BOOTSTRAP")
    print("=" * 100)
    print("Speakers :", len(all_speakers))
    print("Replicates:", cfg.N_BOOT)
    print("Seed     :", cfg.SEED)
    print(
        "Method   : sample speakers with replacement; retain trials "
        "touching sampled speakers; unit trial weight"
    )

    boot = bootstrap_speaker_aware(
        df=df,
        n_boot=cfg.N_BOOT,
        seed=cfg.SEED,
    )

    boot_summary = summarize_bootstrap(boot)

    print("\nBOOTSTRAP SUMMARY")
    print("-" * 100)
    print(
        boot_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.9f}",
        )
    )

    # -------------------------------------------------------------------------
    # Save per-trial audit
    # -------------------------------------------------------------------------
    per_trial_cols = [
        "trial_id",
        "label",
        "enroll_path",
        "test_path",
        "enroll_spk",
        "test_spk",
        "score_b0",
        "score_b6",
        "b0_margin",
        "route_b10",
        "mixed_score",
        "b0_pred",
        "b6_pred",
        "mixed_pred",
        "benefit",
        "harm",
    ]

    per_trial_path = out_dir / "r15_vox1o_per_trial.csv"
    point_path = out_dir / "r15_point_estimates.csv"
    boot_path = out_dir / "r15_bootstrap_replicates.csv"
    boot_summary_path = out_dir / "r15_bootstrap_summary.csv"
    report_path = out_dir / "r15_report.json"

    df[per_trial_cols].to_csv(
        per_trial_path,
        index=False,
    )

    pd.DataFrame(
        [metrics]
    ).to_csv(
        point_path,
        index=False,
    )

    boot.to_csv(
        boot_path,
        index=False,
    )

    boot_summary.to_csv(
        boot_summary_path,
        index=False,
    )

    report = {
        "experiment": (
            "R15 — Held-out frozen-decision expert-benefit validation"
        ),

        "status": "COMPLETED",

        "protocol": {
            "dataset": "VoxCeleb1-O",
            "n_trials": int(len(df)),
            "n_target": int((df["label"] == 1).sum()),
            "n_impostor": int((df["label"] == 0).sum()),
            "n_speakers": int(len(all_speakers)),
            "evaluation_only": True,
        },

        "frozen_parameters": {
            "tau_b0": cfg.TAU_B0,
            "tau_b6": cfg.TAU_B6,
            "tau_b10_mixed": cfg.TAU_B10_MIXED,
            "delta_b10": cfg.DELTA_B10,
        },

        "integrity": {
            "protocol_trial_count_match": True,
            "protocol_label_order_match": True,
            "margin_reconstruction_max_abs_error": float(
                margin_error
            ),
            "route_reconstruction_match": True,
        },

        "point_estimates": metrics,

        "bootstrap": {
            "method": (
                "Speaker resampling with replacement; trials touching at "
                "least one sampled speaker retained once with unit weight."
            ),
            "n_replicates": cfg.N_BOOT,
            "seed": cfg.SEED,
            "summary": boot_summary.to_dict(
                orient="records"
            ),
        },
    }

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\nOUTPUTS")
    print("=" * 100)
    print(per_trial_path)
    print(point_path)
    print(boot_path)
    print(boot_summary_path)
    print(report_path)

    print("\n" + "=" * 100)
    print("R15 COMPLETE")
    print("=" * 100)
    print(
        "No VoxCeleb1-O threshold or routing parameter was optimized."
    )
    print(
        "Benefit/harm is defined relative to the actual frozen B10 "
        "mixed-system decision."
    )


if __name__ == "__main__":
    main()