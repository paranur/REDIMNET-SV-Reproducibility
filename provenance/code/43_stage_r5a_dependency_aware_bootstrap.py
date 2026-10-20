from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score


# ============================================================
# CONFIG
# ============================================================

@dataclass(frozen=True)
class Config:
    project_root: Path = Path(r"package/")

    score_csv: Path = Path(
        r"package/"
        r"\results\stage01_clean_vox1o\scores"
        r"\vox1o_b0_b6_scores.csv"
    )

    expected_score_sha256: str = (
        "f71d954a7e237f8d43140f03df132d88"
        "cabb9e6cbdf5659fd92c94055d38ee4c"
    )

    tau_b0: float = 0.336179674
    delta_b10: float = 0.134631842

    p_target: float = 0.01
    c_miss: float = 1.0
    c_fa: float = 1.0

    n_bootstrap: int = 2000
    seed: int = 20260921

    @property
    def output_dir(self) -> Path:
        return (
            self.project_root
            / "results"
            / "reviewer_audit"
            / "r5a_dependency_aware_bootstrap"
        )


CFG = Config()


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def speaker_from_path(path: str) -> str:
    return str(path).replace("\\", "/").split("/")[0]


def compute_eer(labels, scores):
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

    eer = (
        fpr[idx] + fnr[idx]
    ) / 2.0

    return (
        float(eer * 100.0),
        float(thresholds[idx]),
    )


def compute_min_dcf(labels, scores):
    fpr, tpr, thresholds = roc_curve(
        labels,
        scores,
        pos_label=1,
    )

    fnr = 1.0 - tpr

    c_det = (
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

    idx = int(
        np.argmin(c_det)
    )

    return (
        float(c_det[idx] / c_default),
        float(thresholds[idx]),
    )


def evaluate(labels, scores):
    eer, eer_thr = compute_eer(
        labels,
        scores,
    )

    mindcf, dcf_thr = compute_min_dcf(
        labels,
        scores,
    )

    auc = roc_auc_score(
        labels,
        scores,
    )

    return {
        "eer_pct": float(eer),
        "eer_threshold": float(eer_thr),
        "mindcf": float(mindcf),
        "mindcf_threshold": float(dcf_thr),
        "auc": float(auc),
    }


def percentile_ci(x, alpha=0.05):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    lo = float(
        np.quantile(
            x,
            alpha / 2.0,
        )
    )

    hi = float(
        np.quantile(
            x,
            1.0 - alpha / 2.0,
        )
    )

    return lo, hi


# ============================================================
# WEIGHTED METRICS
# ============================================================

def weighted_roc_points(
    labels,
    scores,
    weights,
):
    """
    Exact weighted ROC construction.

    Used because sklearn roc_curve supports sample_weight,
    but implementing the weighted cumulative counts explicitly
    keeps the bootstrap metric definition transparent.
    """

    labels = np.asarray(
        labels,
        dtype=np.int8,
    )

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    weights = np.asarray(
        weights,
        dtype=np.float64,
    )

    mask = weights > 0

    labels = labels[mask]
    scores = scores[mask]
    weights = weights[mask]

    if len(labels) == 0:
        raise RuntimeError(
            "No positive-weight trials."
        )

    pos_weight = float(
        weights[
            labels == 1
        ].sum()
    )

    neg_weight = float(
        weights[
            labels == 0
        ].sum()
    )

    if (
        pos_weight <= 0
        or neg_weight <= 0
    ):
        raise RuntimeError(
            "Bootstrap replicate lacks "
            "one of the two classes."
        )

    order = np.argsort(
        -scores,
        kind="mergesort",
    )

    y = labels[order]
    s = scores[order]
    w = weights[order]

    tp = np.cumsum(
        w * (y == 1)
    )

    fp = np.cumsum(
        w * (y == 0)
    )

    # Keep only the last item of each tied score.
    distinct = np.r_[
        s[:-1] != s[1:],
        True,
    ]

    tp = tp[distinct]
    fp = fp[distinct]
    thresholds = s[distinct]

    tpr = tp / pos_weight
    fpr = fp / neg_weight

    # Explicit origin.
    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]
    thresholds = np.r_[
        np.inf,
        thresholds,
    ]

    return (
        fpr,
        tpr,
        thresholds,
    )


def weighted_eer(
    labels,
    scores,
    weights,
):
    fpr, tpr, thresholds = (
        weighted_roc_points(
            labels,
            scores,
            weights,
        )
    )

    fnr = 1.0 - tpr

    idx = int(
        np.argmin(
            np.abs(fnr - fpr)
        )
    )

    eer = (
        fpr[idx] + fnr[idx]
    ) / 2.0

    return float(
        eer * 100.0
    )


def weighted_min_dcf(
    labels,
    scores,
    weights,
):
    fpr, tpr, thresholds = (
        weighted_roc_points(
            labels,
            scores,
            weights,
        )
    )

    fnr = 1.0 - tpr

    c_det = (
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

    return float(
        np.min(c_det)
        / c_default
    )


# ============================================================
# MAIN
# ============================================================

def main():
    CFG.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print(
        "STAGE R5-A — DEPENDENCY-AWARE "
        "SPEAKER-CLUSTER BOOTSTRAP"
    )
    print("=" * 100)

    # --------------------------------------------------------
    # INPUT INTEGRITY
    # --------------------------------------------------------

    if not CFG.score_csv.exists():
        raise FileNotFoundError(
            CFG.score_csv
        )

    score_hash = sha256_file(
        CFG.score_csv
    )

    print(
        f"Score CSV SHA256 : {score_hash}"
    )

    if (
        score_hash
        != CFG.expected_score_sha256
    ):
        raise RuntimeError(
            "Canonical score SHA256 mismatch."
        )

    print(
        "Score integrity  : PASS"
    )

    df = pd.read_csv(
        CFG.score_csv
    )

    required = {
        "trial_id",
        "label",
        "enroll_path",
        "test_path",
        "score_b0",
        "score_b6",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"Missing columns: {missing}"
        )

    if len(df) != 37720:
        raise RuntimeError(
            f"Expected 37720 trials, "
            f"got {len(df)}"
        )

    labels = (
        df["label"]
        .to_numpy(
            dtype=np.int8
        )
    )

    score_b0 = (
        df["score_b0"]
        .to_numpy(
            dtype=np.float64
        )
    )

    score_b6 = (
        df["score_b6"]
        .to_numpy(
            dtype=np.float64
        )
    )

    n_target = int(
        (labels == 1).sum()
    )

    n_impostor = int(
        (labels == 0).sum()
    )

    print(
        f"Trials           : {len(df)}"
    )

    print(
        f"Target           : {n_target}"
    )

    print(
        f"Impostor         : {n_impostor}"
    )

    if (
        n_target != 18860
        or n_impostor != 18860
    ):
        raise RuntimeError(
            "Class-balance gate FAILED."
        )

    print(
        "Trial integrity  : PASS"
    )

    # --------------------------------------------------------
    # SPEAKER ENDPOINTS
    # --------------------------------------------------------

    enroll_spk = (
        df["enroll_path"]
        .map(speaker_from_path)
        .to_numpy()
    )

    test_spk = (
        df["test_path"]
        .map(speaker_from_path)
        .to_numpy()
    )

    speakers = sorted(
        set(enroll_spk)
        | set(test_spk)
    )

    n_speakers = len(
        speakers
    )

    print(
        f"Unique speakers  : {n_speakers}"
    )

    if n_speakers != 40:
        raise RuntimeError(
            f"Expected 40 VoxCeleb1-O speakers, "
            f"got {n_speakers}"
        )

    spk_to_idx = {
        spk: i
        for i, spk
        in enumerate(speakers)
    }

    enroll_idx = np.asarray(
        [
            spk_to_idx[x]
            for x in enroll_spk
        ],
        dtype=np.int16,
    )

    test_idx = np.asarray(
        [
            spk_to_idx[x]
            for x in test_spk
        ],
        dtype=np.int16,
    )

    print(
        "Speaker gate     : PASS"
    )

    # --------------------------------------------------------
    # FROZEN-B10 REPRODUCTION
    # --------------------------------------------------------

    route_b10 = (
        np.abs(
            score_b0
            - CFG.tau_b0
        )
        <= CFG.delta_b10
    )

    score_b10 = np.where(
        route_b10,
        score_b6,
        score_b0,
    )

    usage = float(
        route_b10.mean()
        * 100.0
    )

    static_b6 = evaluate(
        labels,
        score_b6,
    )

    frozen_b10 = evaluate(
        labels,
        score_b10,
    )

    print()
    print("=" * 100)
    print("CANONICAL REPRODUCTION")
    print("=" * 100)

    print(
        f"B10 usage : {usage:.9f}%"
    )

    print(
        f"b6 EER    : "
        f"{static_b6['eer_pct']:.9f}%"
    )

    print(
        f"b6 minDCF : "
        f"{static_b6['mindcf']:.9f}"
    )

    print(
        f"B10 EER   : "
        f"{frozen_b10['eer_pct']:.9f}%"
    )

    print(
        f"B10 minDCF: "
        f"{frozen_b10['mindcf']:.9f}"
    )

    # Exact historical gates with small numeric tolerance.
    gates = {
        "usage": abs(
            usage
            - 9.652704136
        ) <= 1e-6,

        "b6_eer": abs(
            static_b6["eer_pct"]
            - 0.715800636
        ) <= 1e-6,

        "b6_mindcf": abs(
            static_b6["mindcf"]
            - 0.054347826
        ) <= 1e-6,

        "b10_eer": abs(
            frozen_b10["eer_pct"]
            - 0.723753977
        ) <= 1e-6,

        "b10_mindcf": abs(
            frozen_b10["mindcf"]
            - 0.135206787
        ) <= 1e-6,
    }

    print()
    print(
        f"Reproduction gates: {gates}"
    )

    if not all(
        gates.values()
    ):
        raise RuntimeError(
            "Canonical reproduction FAILED."
        )

    print(
        "Canonical reproduction: PASS"
    )

    # --------------------------------------------------------
    # DEPENDENCY-AWARE PAIRED BOOTSTRAP
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "PAIRED SPEAKER-CLUSTER BOOTSTRAP"
    )
    print("=" * 100)

    print(
        f"Bootstrap replicates : "
        f"{CFG.n_bootstrap}"
    )

    print(
        f"Bootstrap seed       : "
        f"{CFG.seed}"
    )

    print(
        "Resampling unit      : speaker"
    )

    print(
        "Dyadic trial weight  : "
        "multiplicity(enroll) * "
        "multiplicity(test)"
    )

    rng = np.random.default_rng(
        CFG.seed
    )

    rows = []

    skipped = 0

    for b in range(
        CFG.n_bootstrap
    ):

        sampled = rng.integers(
            0,
            n_speakers,
            size=n_speakers,
        )

        multiplicity = np.bincount(
            sampled,
            minlength=n_speakers,
        ).astype(
            np.float64
        )

        weights = (
            multiplicity[enroll_idx]
            * multiplicity[test_idx]
        )

        try:
            b6_eer = weighted_eer(
                labels,
                score_b6,
                weights,
            )

            b10_eer = weighted_eer(
                labels,
                score_b10,
                weights,
            )

            b6_dcf = weighted_min_dcf(
                labels,
                score_b6,
                weights,
            )

            b10_dcf = weighted_min_dcf(
                labels,
                score_b10,
                weights,
            )

        except RuntimeError:
            skipped += 1
            continue

        rows.append({
            "replicate": b,

            "b6_eer_pct":
                b6_eer,

            "b10_eer_pct":
                b10_eer,

            "delta_eer_pp_b10_minus_b6":
                b10_eer - b6_eer,

            "b6_mindcf":
                b6_dcf,

            "b10_mindcf":
                b10_dcf,

            "delta_mindcf_b10_minus_b6":
                b10_dcf - b6_dcf,

            "effective_trial_weight":
                float(
                    weights.sum()
                ),

            "selected_unique_speakers":
                int(
                    np.sum(
                        multiplicity > 0
                    )
                ),
        })

        if (
            (b + 1) % 100 == 0
            or b + 1
            == CFG.n_bootstrap
        ):
            print(
                f"Bootstrap: "
                f"{b + 1}/"
                f"{CFG.n_bootstrap}"
            )

    boot = pd.DataFrame(
        rows
    )

    if len(boot) < (
        CFG.n_bootstrap * 0.95
    ):
        raise RuntimeError(
            "Too many invalid bootstrap "
            "replicates."
        )

    print(
        f"Valid replicates     : "
        f"{len(boot)}"
    )

    print(
        f"Skipped replicates   : "
        f"{skipped}"
    )

    # --------------------------------------------------------
    # CONFIDENCE INTERVALS
    # --------------------------------------------------------

    metrics = {}

    columns = [
        "b6_eer_pct",
        "b10_eer_pct",
        "delta_eer_pp_b10_minus_b6",
        "b6_mindcf",
        "b10_mindcf",
        "delta_mindcf_b10_minus_b6",
    ]

    for col in columns:
        values = (
            boot[col]
            .to_numpy(
                dtype=np.float64
            )
        )

        lo, hi = percentile_ci(
            values
        )

        metrics[col] = {
            "bootstrap_mean":
                float(
                    values.mean()
                ),

            "bootstrap_sd":
                float(
                    values.std(
                        ddof=1
                    )
                ),

            "ci95_low":
                lo,

            "ci95_high":
                hi,
        }

    print()
    print("=" * 100)
    print("R5-A RESULTS")
    print("=" * 100)

    print()
    print("EER")

    print(
        "b6 95% CI  : "
        f"[{metrics['b6_eer_pct']['ci95_low']:.6f}, "
        f"{metrics['b6_eer_pct']['ci95_high']:.6f}] %"
    )

    print(
        "B10 95% CI : "
        f"[{metrics['b10_eer_pct']['ci95_low']:.6f}, "
        f"{metrics['b10_eer_pct']['ci95_high']:.6f}] %"
    )

    print(
        "ΔEER B10-b6 95% CI: "
        f"[{metrics['delta_eer_pp_b10_minus_b6']['ci95_low']:+.6f}, "
        f"{metrics['delta_eer_pp_b10_minus_b6']['ci95_high']:+.6f}] pp"
    )

    print()
    print("minDCF")

    print(
        "b6 95% CI  : "
        f"[{metrics['b6_mindcf']['ci95_low']:.6f}, "
        f"{metrics['b6_mindcf']['ci95_high']:.6f}]"
    )

    print(
        "B10 95% CI : "
        f"[{metrics['b10_mindcf']['ci95_low']:.6f}, "
        f"{metrics['b10_mindcf']['ci95_high']:.6f}]"
    )

    print(
        "ΔminDCF B10-b6 95% CI: "
        f"[{metrics['delta_mindcf_b10_minus_b6']['ci95_low']:+.6f}, "
        f"{metrics['delta_mindcf_b10_minus_b6']['ci95_high']:+.6f}]"
    )

    # --------------------------------------------------------
    # ZERO-CROSSING DIAGNOSTIC
    # --------------------------------------------------------

    eer_delta = metrics[
        "delta_eer_pp_b10_minus_b6"
    ]

    dcf_delta = metrics[
        "delta_mindcf_b10_minus_b6"
    ]

    eer_crosses_zero = (
        eer_delta["ci95_low"] <= 0.0
        <= eer_delta["ci95_high"]
    )

    dcf_crosses_zero = (
        dcf_delta["ci95_low"] <= 0.0
        <= dcf_delta["ci95_high"]
    )

    print()
    print("=" * 100)
    print("INTERPRETATION GUARDRAIL")
    print("=" * 100)

    print(
        "ΔEER CI contains zero    : "
        f"{eer_crosses_zero}"
    )

    print(
        "ΔminDCF CI contains zero : "
        f"{dcf_crosses_zero}"
    )

    print()
    print(
        "IMPORTANT: A confidence interval "
        "containing zero is NOT evidence "
        "of equivalence."
    )

    print(
        "No equivalence or non-inferiority "
        "claim is made in R5-A."
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    bootstrap_csv = (
        CFG.output_dir
        / "r5a_speaker_cluster_bootstrap_replicates.csv"
    )

    summary_csv = (
        CFG.output_dir
        / "r5a_dependency_aware_ci_summary.csv"
    )

    report_json = (
        CFG.output_dir
        / "r5a_dependency_aware_bootstrap_report.json"
    )

    boot.to_csv(
        bootstrap_csv,
        index=False,
    )

    summary_rows = []

    for metric, values in metrics.items():
        summary_rows.append({
            "metric":
                metric,

            **values,
        })

    pd.DataFrame(
        summary_rows
    ).to_csv(
        summary_csv,
        index=False,
    )

    report = {
        "stage":
            "R5-A",

        "status":
            "COMPLETE_NOT_YET_LOCKED",

        "input": {
            "score_csv":
                str(
                    CFG.score_csv
                ),

            "sha256":
                score_hash,

            "trials":
                len(df),

            "target":
                n_target,

            "impostor":
                n_impostor,

            "speakers":
                n_speakers,
        },

        "frozen_b10": {
            "tau_b0":
                CFG.tau_b0,

            "delta_b10":
                CFG.delta_b10,

            "usage_pct":
                usage,
        },

        "canonical": {
            "b6":
                static_b6,

            "b10":
                frozen_b10,
        },

        "bootstrap": {
            "method":
                "paired speaker-cluster bootstrap "
                "with dyadic endpoint multiplicity weighting",

            "n_requested":
                CFG.n_bootstrap,

            "n_valid":
                len(boot),

            "n_skipped":
                skipped,

            "seed":
                CFG.seed,

            "speaker_count":
                n_speakers,

            "trial_weight":
                "m_enroll * m_test",
        },

        "ci95":
            metrics,

        "zero_crossing": {
            "delta_eer":
                eer_crosses_zero,

            "delta_mindcf":
                dcf_crosses_zero,
        },

        "guardrail":
            (
                "CI overlap or a paired-difference "
                "CI containing zero must not be "
                "interpreted as equivalence. "
                "R5-A performs uncertainty estimation, "
                "not an equivalence/non-inferiority test."
            ),
    }

    with open(
        report_json,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 100)
    print("ARTIFACTS")
    print("=" * 100)

    print(
        bootstrap_csv
    )

    print(
        summary_csv
    )

    print(
        report_json
    )

    print()
    print("=" * 100)
    print(
        "STAGE R5-A COMPLETE — "
        "NOT YET LOCKED"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()