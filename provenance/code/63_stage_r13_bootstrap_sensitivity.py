from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


@dataclass(frozen=True)
class Config:
    score_csv: Path = Path(
        r"package/"
        r"\results\stage01_clean_vox1o\scores"
        r"\vox1o_b0_b6_scores.csv"
    )

    output_dir: Path = Path(
        r"package/"
        r"\results\reviewer_audit"
        r"\r13_bootstrap_sensitivity"
    )

    tau_b0: float = 0.336179674
    delta_b10: float = 0.134631842

    p_target: float = 0.01
    c_miss: float = 1.0
    c_fa: float = 1.0

    n_bootstrap: int = 2000
    seed: int = 20260921


CFG = Config()


def speaker_from_path(path):
    return str(path).replace("\\", "/").split("/")[0]


def eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    i = int(np.argmin(np.abs(fnr - fpr)))
    return float(100.0 * (fpr[i] + fnr[i]) / 2.0)


def min_dcf(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr

    c_det = (
        CFG.c_miss * CFG.p_target * fnr
        + CFG.c_fa * (1.0 - CFG.p_target) * fpr
    )

    c_default = min(
        CFG.c_miss * CFG.p_target,
        CFG.c_fa * (1.0 - CFG.p_target),
    )

    return float(np.min(c_det) / c_default)


def ci95(x):
    x = np.asarray(x, dtype=float)
    return (
        float(np.quantile(x, 0.025)),
        float(np.quantile(x, 0.975)),
    )


def main():

    CFG.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("R13 — ALTERNATIVE SPEAKER-AWARE BOOTSTRAP SENSITIVITY")
    print("=" * 90)

    df = pd.read_csv(CFG.score_csv)

    required = {
        "label",
        "enroll_path",
        "test_path",
        "score_b0",
        "score_b6",
    }

    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns: {sorted(missing)}")

    labels = df["label"].to_numpy(dtype=np.int8)
    b0 = df["score_b0"].to_numpy(dtype=float)
    b6 = df["score_b6"].to_numpy(dtype=float)

    enroll_spk = df["enroll_path"].map(speaker_from_path).to_numpy()
    test_spk = df["test_path"].map(speaker_from_path).to_numpy()

    speakers = sorted(set(enroll_spk) | set(test_spk))

    if len(speakers) != 40:
        raise RuntimeError(
            f"Expected 40 speakers, got {len(speakers)}"
        )

    # Canonical frozen B10.
    route = np.abs(b0 - CFG.tau_b0) <= CFG.delta_b10
    b10 = np.where(route, b6, b0)

    print(f"Trials       : {len(df):,}")
    print(f"Speakers     : {len(speakers)}")
    print(f"B10 usage    : {100 * route.mean():.6f}%")
    print(f"Replicates   : {CFG.n_bootstrap}")
    print(f"Seed         : {CFG.seed}")
    print()
    print(
        "Alternative bootstrap: resample speakers, retain trials touching "
        "a sampled speaker, and assign each retained trial unit weight."
    )

    rng = np.random.default_rng(CFG.seed)

    rows = []
    skipped = 0

    for r in range(CFG.n_bootstrap):

        sampled = rng.choice(
            speakers,
            size=len(speakers),
            replace=True,
        )

        selected = set(sampled)

        # Alternative to m_enroll * m_test:
        # no multiplicity weighting.
        mask = (
            np.isin(enroll_spk, list(selected))
            | np.isin(test_spk, list(selected))
        )

        y = labels[mask]

        if (
            len(y) == 0
            or np.sum(y == 1) == 0
            or np.sum(y == 0) == 0
        ):
            skipped += 1
            continue

        s6 = b6[mask]
        s10 = b10[mask]

        b6_eer = eer(y, s6)
        b10_eer = eer(y, s10)

        b6_dcf = min_dcf(y, s6)
        b10_dcf = min_dcf(y, s10)

        rows.append({
            "replicate": r,
            "n_trials": int(mask.sum()),
            "n_unique_sampled_speakers": len(selected),

            "b6_eer_pct": b6_eer,
            "b10_eer_pct": b10_eer,
            "delta_eer_pp_b10_minus_b6":
                b10_eer - b6_eer,

            "b6_mindcf": b6_dcf,
            "b10_mindcf": b10_dcf,
            "delta_mindcf_b10_minus_b6":
                b10_dcf - b6_dcf,
        })

        if (r + 1) % 200 == 0:
            print(f"Bootstrap: {r + 1}/{CFG.n_bootstrap}")

    boot = pd.DataFrame(rows)

    if len(boot) < 0.95 * CFG.n_bootstrap:
        raise RuntimeError("Too many invalid bootstrap replicates.")

    summary = []

    for col in [
        "b6_eer_pct",
        "b10_eer_pct",
        "delta_eer_pp_b10_minus_b6",
        "b6_mindcf",
        "b10_mindcf",
        "delta_mindcf_b10_minus_b6",
    ]:
        x = boot[col].to_numpy(dtype=float)
        lo, hi = ci95(x)

        summary.append({
            "metric": col,
            "mean": float(x.mean()),
            "sd": float(x.std(ddof=1)),
            "ci95_low": lo,
            "ci95_high": hi,
        })

    summary_df = pd.DataFrame(summary)

    print()
    print("=" * 90)
    print("R13 RESULTS")
    print("=" * 90)

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.9f}",
        )
    )

    eer_row = summary_df[
        summary_df["metric"]
        == "delta_eer_pp_b10_minus_b6"
    ].iloc[0]

    dcf_row = summary_df[
        summary_df["metric"]
        == "delta_mindcf_b10_minus_b6"
    ].iloc[0]

    eer_zero = (
        eer_row["ci95_low"] <= 0 <= eer_row["ci95_high"]
    )

    dcf_zero = (
        dcf_row["ci95_low"] <= 0 <= dcf_row["ci95_high"]
    )

    print()
    print(
        "Delta EER 95% CI contains zero   :",
        eer_zero,
    )
    print(
        "Delta minDCF 95% CI contains zero:",
        dcf_zero,
    )

    boot_path = (
        CFG.output_dir
        / "r13_bootstrap_replicates.csv"
    )

    summary_path = (
        CFG.output_dir
        / "r13_bootstrap_summary.csv"
    )

    report_path = (
        CFG.output_dir
        / "r13_bootstrap_report.json"
    )

    boot.to_csv(boot_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    report = {
        "stage": "R13",
        "method": (
            "speaker-resampling sensitivity analysis with "
            "unit-weight retained trials; no m_i*m_j weighting"
        ),
        "n_bootstrap": CFG.n_bootstrap,
        "seed": CFG.seed,
        "valid_replicates": len(boot),
        "skipped_replicates": skipped,
        "delta_eer_ci_contains_zero": bool(eer_zero),
        "delta_mindcf_ci_contains_zero": bool(dcf_zero),
        "summary": summary_df.to_dict(orient="records"),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print()
    print("Artifacts:")
    print(boot_path)
    print(summary_path)
    print(report_path)

    print()
    print("=" * 90)
    print("R13 COMPLETE — NOT YET LOCKED")
    print("=" * 90)


if __name__ == "__main__":
    main()