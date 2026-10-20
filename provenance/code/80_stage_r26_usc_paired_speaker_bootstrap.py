from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "reviewer_audit"
    / "r19e_usc_test_final"
    / "usc_test_final_frozen_scores.csv"
)

OUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "reviewer_audit"
    / "r26_usc_paired_speaker_bootstrap"
)

BOOT_CSV = OUT_DIR / "r26_bootstrap_samples.csv"
SUMMARY_JSON = OUT_DIR / "r26_summary.json"

N_BOOT = 2000
SEED = 20261007

PTARGET = 0.01
C_MISS = 1.0
C_FA = 1.0


def compute_eer(labels, scores):
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)

    fpr, tpr, _ = roc_curve(
        labels,
        scores,
        drop_intermediate=False,
    )

    fnr = 1.0 - tpr

    i = int(
        np.nanargmin(
            np.abs(fpr - fnr)
        )
    )

    return 100.0 * float(
        (fpr[i] + fnr[i]) / 2.0
    )


def compute_min_dcf(labels, scores):
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)

    order = np.argsort(scores)[::-1]
    y = labels[order]

    n_tar = int(np.sum(y == 1))
    n_non = int(np.sum(y == 0))

    if n_tar == 0 or n_non == 0:
        return np.nan

    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)

    fn = n_tar - tp

    pmiss = fn / n_tar
    pfa = fp / n_non

    cost = (
        C_MISS * PTARGET * pmiss
        + C_FA * (1.0 - PTARGET) * pfa
    )

    # Include reject-all operating point.
    best = min(
        float(np.min(cost)),
        C_MISS * PTARGET,
    )

    default = min(
        C_MISS * PTARGET,
        C_FA * (1.0 - PTARGET),
    )

    return best / default


def percentile_ci(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]

    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "ci95_low": float(np.percentile(x, 2.5)),
        "ci95_high": float(np.percentile(x, 97.5)),
    }


def main():
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print("R26 — USC-SV SPEAKER-AWARE PAIRED BOOTSTRAP")
    print("=" * 100)

    df = pd.read_csv(INPUT_CSV)

    required = {
        "label",
        "claimed_speaker",
        "query_speaker",
        "score_b0",
        "score_b6",
        "score_frozen_b10",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing columns: {missing}"
        )

    if len(df) != 5390:
        raise RuntimeError(
            f"Expected 5390 TEST trials, found {len(df)}"
        )

    # ------------------------------------------------------------------
    # Protocol audit
    # ------------------------------------------------------------------

    target = df["label"] == 1
    impostor = df["label"] == 0

    if not np.all(
        df.loc[target, "claimed_speaker"].to_numpy()
        ==
        df.loc[target, "query_speaker"].to_numpy()
    ):
        raise RuntimeError(
            "Target identity consistency failed."
        )

    if not np.all(
        df.loc[impostor, "claimed_speaker"].to_numpy()
        !=
        df.loc[impostor, "query_speaker"].to_numpy()
    ):
        raise RuntimeError(
            "Impostor identity consistency failed."
        )

    speakers = sorted(
        set(df["claimed_speaker"].tolist())
        | set(df["query_speaker"].tolist())
    )

    print("Trials       :", len(df))
    print("Targets      :", int(target.sum()))
    print("Impostors    :", int(impostor.sum()))
    print("TEST speakers:", len(speakers))
    print("Bootstrap N  :", N_BOOT)
    print("Seed         :", SEED)

    if len(speakers) != 49:
        raise RuntimeError(
            f"Expected 49 TEST speakers, found {len(speakers)}"
        )

    labels_full = df["label"].to_numpy(dtype=np.int32)

    score_columns = {
        "b0": "score_b0",
        "b6": "score_b6",
        "B10": "score_frozen_b10",
    }

    # ------------------------------------------------------------------
    # Original full TEST metrics
    # ------------------------------------------------------------------

    original = {}

    print()
    print("=" * 100)
    print("ORIGINAL TEST")
    print("=" * 100)

    for name, col in score_columns.items():
        scores = df[col].to_numpy(dtype=np.float64)

        e = compute_eer(
            labels_full,
            scores,
        )

        d = compute_min_dcf(
            labels_full,
            scores,
        )

        original[name] = {
            "eer_percent": e,
            "minDCF": d,
        }

        print(
            f"{name:>4s} | "
            f"EER={e:.6f}% | "
            f"minDCF={d:.9f}"
        )

    # ------------------------------------------------------------------
    # Speaker-aware paired bootstrap
    #
    # Each sampled speaker occurrence contributes all trials touching
    # that speaker. Multiplicity is retained. This preserves speaker
    # dependence rather than treating the 5390 trials as IID.
    #
    # The SAME bootstrap sample is used for b0, b6, and B10.
    # ------------------------------------------------------------------

    speaker_masks = {}

    claimed = df[
        "claimed_speaker"
    ].to_numpy()

    query = df[
        "query_speaker"
    ].to_numpy()

    for spk in speakers:
        speaker_masks[spk] = (
            (claimed == spk)
            | (query == spk)
        )

    rng = np.random.default_rng(SEED)

    rows = []

    print()
    print("=" * 100)
    print("BOOTSTRAP")
    print("=" * 100)

    for b in range(N_BOOT):

        sampled = rng.choice(
            speakers,
            size=len(speakers),
            replace=True,
        )

        # Retain multiplicity of sampled speakers.
        weights = np.zeros(
            len(df),
            dtype=np.int16,
        )

        for spk in sampled:
            weights += speaker_masks[spk].astype(
                np.int16
            )

        idx = np.repeat(
            np.arange(len(df)),
            weights,
        )

        y = labels_full[idx]

        # Defensive check: every bootstrap sample should contain both
        # target and impostor trials.
        if (
            np.sum(y == 1) == 0
            or np.sum(y == 0) == 0
        ):
            continue

        metrics = {}

        for name, col in score_columns.items():

            s = (
                df[col]
                .to_numpy(dtype=np.float64)[idx]
            )

            metrics[name] = {
                "eer": compute_eer(
                    y,
                    s,
                ),
                "dcf": compute_min_dcf(
                    y,
                    s,
                ),
            }

        rows.append(
            {
                "bootstrap_id": b,

                "n_effective_rows":
                    int(len(idx)),

                "n_target":
                    int(np.sum(y == 1)),

                "n_impostor":
                    int(np.sum(y == 0)),

                "eer_b0":
                    metrics["b0"]["eer"],

                "eer_b6":
                    metrics["b6"]["eer"],

                "eer_B10":
                    metrics["B10"]["eer"],

                "delta_eer_B10_minus_b0":
                    metrics["B10"]["eer"]
                    - metrics["b0"]["eer"],

                "delta_eer_B10_minus_b6":
                    metrics["B10"]["eer"]
                    - metrics["b6"]["eer"],

                "delta_eer_b6_minus_b0":
                    metrics["b6"]["eer"]
                    - metrics["b0"]["eer"],

                "minDCF_b0":
                    metrics["b0"]["dcf"],

                "minDCF_b6":
                    metrics["b6"]["dcf"],

                "minDCF_B10":
                    metrics["B10"]["dcf"],

                "delta_minDCF_B10_minus_b0":
                    metrics["B10"]["dcf"]
                    - metrics["b0"]["dcf"],

                "delta_minDCF_B10_minus_b6":
                    metrics["B10"]["dcf"]
                    - metrics["b6"]["dcf"],

                "delta_minDCF_b6_minus_b0":
                    metrics["b6"]["dcf"]
                    - metrics["b0"]["dcf"],
            }
        )

        if (b + 1) % 200 == 0:
            print(
                f"Bootstrap: "
                f"{b + 1}/{N_BOOT}"
            )

    boot = pd.DataFrame(rows)

    if len(boot) != N_BOOT:
        raise RuntimeError(
            f"Expected {N_BOOT} valid bootstrap "
            f"samples, obtained {len(boot)}"
        )

    boot.to_csv(
        BOOT_CSV,
        index=False,
    )

    # ------------------------------------------------------------------
    # Paired uncertainty summaries
    # ------------------------------------------------------------------

    quantities = [
        "eer_b0",
        "eer_b6",
        "eer_B10",
        "delta_eer_B10_minus_b0",
        "delta_eer_B10_minus_b6",
        "delta_eer_b6_minus_b0",
        "minDCF_b0",
        "minDCF_b6",
        "minDCF_B10",
        "delta_minDCF_B10_minus_b0",
        "delta_minDCF_B10_minus_b6",
        "delta_minDCF_b6_minus_b0",
    ]

    uncertainty = {
        q: percentile_ci(
            boot[q].to_numpy()
        )
        for q in quantities
    }

    print()
    print("=" * 100)
    print("PAIRED DIFFERENCE 95% CIs")
    print("=" * 100)

    for q in [
        "delta_eer_B10_minus_b0",
        "delta_eer_B10_minus_b6",
        "delta_eer_b6_minus_b0",
        "delta_minDCF_B10_minus_b0",
        "delta_minDCF_B10_minus_b6",
        "delta_minDCF_b6_minus_b0",
    ]:
        x = uncertainty[q]

        print(
            f"{q}: "
            f"mean={x['mean']:.9f} | "
            f"95% CI "
            f"[{x['ci95_low']:.9f}, "
            f"{x['ci95_high']:.9f}]"
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    summary = {
        "experiment":
            "R26 USC-SV speaker-aware paired bootstrap",

        "status":
            "COMPLETE_PENDING_SCIENTIFIC_LOCK",

        "input":
            str(INPUT_CSV),

        "protocol": {
            "trials":
                int(len(df)),
            "targets":
                int(target.sum()),
            "impostors":
                int(impostor.sum()),
            "test_speakers":
                int(len(speakers)),
        },

        "bootstrap": {
            "unit":
                "speaker",
            "paired_across_systems":
                True,
            "speaker_multiplicity_retained":
                True,
            "trial_rule":
                "all trials touching each sampled speaker",
            "n_bootstrap":
                N_BOOT,
            "seed":
                SEED,
            "ci":
                "percentile 95%",
        },

        "original_test_metrics":
            original,

        "uncertainty":
            uncertainty,
    }

    with SUMMARY_JSON.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("Bootstrap CSV :", BOOT_CSV)
    print("Summary JSON  :", SUMMARY_JSON)

    print()
    print(
        "COMPLETE — speaker-aware paired "
        "uncertainty analysis finished."
    )


if __name__ == "__main__":
    main()