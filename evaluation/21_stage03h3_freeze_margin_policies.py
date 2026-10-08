import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    PROJECT_ROOT / "results" / "stage03_dev_cal"
    / "stage03g_benefit" / "dev_frozen_threshold_benefit_labels.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "stage03_dev_cal"
    / "stage03h_policy_freeze"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUTPUT_DIR / "dev_margin_frozen_policies.csv"
OUT_JSON = OUTPUT_DIR / "dev_margin_frozen_policies.json"
OUT_SHA = OUTPUT_DIR / "dev_margin_frozen_policies.sha256.txt"

BUDGETS = [5, 10, 20, 30]


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 100)
    print("STAGE 03H-3 — DEV MARGIN CUTOFF SELECTION & POLICY FREEZING")
    print("=" * 100)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    df = pd.read_csv(INPUT_CSV)

    required = {"trial_id", "b0_abs_margin"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns: {sorted(missing)}")

    if len(df) != 50000:
        raise RuntimeError(f"Expected 50000 DEV trials, found {len(df)}")
    if df["trial_id"].duplicated().any():
        raise RuntimeError("Duplicate trial_id detected.")
    if not np.isfinite(df["b0_abs_margin"].to_numpy(dtype=float)).all():
        raise RuntimeError("Non-finite b0_abs_margin detected.")

    # Stable deterministic ordering identical in principle to Stage 03H-2:
    # smallest uncertainty margin first, trial_id as tie-break.
    ranked = (
        df[["trial_id", "b0_abs_margin"]]
        .assign(trial_id=lambda x: x["trial_id"].astype(str))
        .sort_values(["b0_abs_margin", "trial_id"], ascending=[True, True])
        .reset_index(drop=True)
    )

    rows = []

    print(f"DEV trials: {len(ranked)}")
    print()

    for budget in BUDGETS:
        n_route = int(round(len(ranked) * budget / 100.0))

        # The operational rule will be margin <= delta.
        # Record the kth margin and audit how many trials this actually selects.
        delta = float(ranked.iloc[n_route - 1]["b0_abs_margin"])
        strict_below = int((ranked["b0_abs_margin"] < delta).sum())
        at_or_below = int((ranked["b0_abs_margin"] <= delta).sum())
        ties_at_cutoff = int((ranked["b0_abs_margin"] == delta).sum())

        actual_usage = 100.0 * at_or_below / len(ranked)

        rows.append({
            "policy_name": f"MARGIN_B{budget}",
            "target_dev_budget_percent": float(budget),
            "target_dev_routed_trials": int(n_route),
            "delta_margin_cutoff": delta,
            "operational_rule": "route_to_b6_if_b0_abs_margin_le_delta",
            "dev_trials_strictly_below_cutoff": strict_below,
            "dev_trials_at_or_below_cutoff": at_or_below,
            "ties_at_cutoff": ties_at_cutoff,
            "actual_dev_usage_percent_with_threshold_rule": actual_usage,
        })

        print(
            f"B={budget:>2}% | target={n_route:>5} | "
            f"delta={delta:.9f} | <=delta={at_or_below:>5} "
            f"({actual_usage:.4f}%) | ties={ties_at_cutoff}"
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    csv_hash = sha256_file(OUT_CSV)

    report = {
        "stage": "03H-3",
        "status": "FROZEN",
        "source": str(INPUT_CSV),
        "source_sha256": sha256_file(INPUT_CSV),
        "dev_trials": int(len(ranked)),
        "routing_score": "b0_abs_margin = abs(score_b0 - CAL_frozen_tau_b0)",
        "deployment_rule": "route to ReDimNet-b6 iff b0_abs_margin <= delta",
        "important": [
            "These delta values are selected on DEV only.",
            "They must not be recomputed, re-ranked, or retuned on VoxCeleb1-O.",
            "Actual b6 usage on an unseen test set is an outcome, not a fixed quota.",
            "CAL-frozen b0 decision threshold remains unchanged."
        ],
        "policies": rows,
        "policy_csv_sha256": csv_hash,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with open(OUT_SHA, "w", encoding="utf-8") as f:
        f.write(f"{csv_hash}  {OUT_CSV.name}\n")

    print()
    print("FROZEN POLICY FILES")
    print(f"CSV   : {OUT_CSV}")
    print(f"JSON  : {OUT_JSON}")
    print(f"SHA256: {OUT_SHA}")
    print(f"Policy CSV SHA256: {csv_hash}")
    print()
    print("IMPORTANT:")
    print("VoxCeleb1-O must use these exact delta values.")
    print("Do NOT select the lowest-margin X% of VoxCeleb1-O trials.")
    print("Test-set b6 usage must be reported as an observed result.")
    print()
    print("PASS — Stage 03H-3 DEV routing policies frozen.")
    print("=" * 100)


if __name__ == "__main__":
    main()
