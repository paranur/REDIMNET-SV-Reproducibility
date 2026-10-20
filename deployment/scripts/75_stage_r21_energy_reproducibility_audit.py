from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

ROOT = Path(".")

R9_CSV = (
    ROOT
    / "results"
    / "reviewer_audit"
    / "r9_at085_energy_benchmark"
    / "r9_at085_energy_summary.csv"
)

R14C_CSV = (
    ROOT
    / "results"
    / "reviewer_audit"
    / "r14c_pi5_b2_at085_energy"
    / "r14c_b2_energy_summary.csv"
)

OUT_DIR = (
    ROOT
    / "results"
    / "reviewer_audit"
    / "r21_energy_reproducibility"
)

OUT_RUN_AUDIT = OUT_DIR / "r21_energy_run_validity.csv"
OUT_SUMMARY = OUT_DIR / "r21_energy_valid_summary.csv"
OUT_JSON = OUT_DIR / "r21_energy_reproducibility.json"


# =============================================================================
# FROZEN VALIDITY DECISION
# =============================================================================

# IMPORTANT:
# R9 STATIC_B0 run 2 is NOT excluded because its measured value is high.
#
# During that run, the cumulative end-energy reading was not recorded at
# the prescribed end-of-benchmark boundary. Therefore the recorded energy
# interval included time outside the defined benchmark interval.
#
# Run 4 was deliberately performed as a replacement measurement.
#
# Historical raw data remain untouched.

INVALID_R9 = {
    ("STATIC_B0", 2): (
        "INVALID_PROTOCOL: cumulative end-energy reading was not recorded "
        "at the prescribed end-of-run boundary; run 4 was performed as "
        "the replacement repetition."
    )
}

EXPECTED_VALID_R9 = {
    "STATIC_B0": {1, 3, 4},
    "STATIC_B6": {1, 2, 3},
    "SEQUENTIAL_ALL": {1, 2, 3},
    "FROZEN_B10": {1, 2, 3},
}

EXPECTED_VALID_R14C = {
    "STATIC_B2": {1, 2, 3},
}


# =============================================================================
# HELPERS
# =============================================================================

def check(condition, message):
    if not condition:
        raise RuntimeError(f"FAIL — {message}")
    print(f"PASS — {message}")


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


def mean_sd(values):
    x = np.asarray(values, dtype=float)

    if len(x) < 2:
        return float(np.mean(x)), float("nan")

    return (
        float(np.mean(x)),
        float(np.std(x, ddof=1)),
    )


# =============================================================================
# HISTORICAL R9 CSV LOADER
# =============================================================================

def load_r9_historical_csv(path):
    """
    Read the preserved R9 CSV without modifying it.

    Historical issue:
    - initial rows used the original 25-field schema;
    - later rows added four metadata fields:
        pi_temp_start_c
        pi_temp_end_c
        pi_temp_delta_c
        embedding_cache_mode
    - the original CSV header itself was not expanded.

    Therefore:
    - 25-field rows are expanded in memory;
    - 29-field rows are read directly;
    - raw historical CSV is never rewritten.
    """

    columns = [
        "timestamp",
        "pipeline",
        "run_number",
        "trials",
        "target",
        "impostor",
        "start_wh",
        "end_wh",
        "energy_wh",
        "energy_j",
        "energy_j_per_trial",
        "wall_seconds",
        "average_system_power_w",
        "benchmark_mean_latency_ms",
        "benchmark_median_latency_ms",
        "benchmark_p95_latency_ms",
        "benchmark_p99_latency_ms",
        "eer_percent",
        "min_dcf",
        "auc",
        "routed_count",
        "b6_usage_percent",
        "pi_temp_start_c",
        "pi_temp_end_c",
        "pi_temp_delta_c",
        "embedding_cache_mode",
        "manifest_sha256",
        "tau_b0",
        "delta_b10",
    ]

    with open(path, "r", encoding="utf-8") as f:
        lines = [
            line.rstrip("\n")
            for line in f
            if line.strip()
        ]

    check(
        len(lines) >= 2,
        "R9 historical CSV is non-empty",
    )

    original_header = lines[0].split(",")

    check(
        len(original_header) == 25,
        "R9 historical header contains original 25 fields",
    )

    rows = []

    count_original = 0
    count_extended = 0

    for line_number, line in enumerate(
        lines[1:],
        start=2,
    ):
        fields = line.split(",")

        if len(fields) == 25:
            # Original layout:
            #
            # indices 0..21:
            # through b6_usage_percent
            #
            # indices 22..24:
            # manifest_sha256
            # tau_b0
            # delta_b10
            #
            # Insert the four later metadata fields in memory.

            fields = (
                fields[:22]
                + ["", "", "", ""]
                + fields[22:]
            )

            count_original += 1

        elif len(fields) == 29:
            count_extended += 1

        else:
            raise RuntimeError(
                "FAIL — unexpected R9 field count "
                f"at line {line_number}: "
                f"{len(fields)} fields"
            )

        if len(fields) != len(columns):
            raise RuntimeError(
                "FAIL — reconstructed R9 row has "
                f"{len(fields)} fields instead of "
                f"{len(columns)} at line {line_number}"
            )

        rows.append(fields)

    df = pd.DataFrame(
        rows,
        columns=columns,
    )

    string_columns = {
        "timestamp",
        "pipeline",
        "embedding_cache_mode",
        "manifest_sha256",
    }

    for col in columns:
        if col not in string_columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    print()
    print("R9 HISTORICAL SCHEMA")
    print("-" * 112)
    print(
        f"Original-schema rows : {count_original}"
    )
    print(
        f"Extended-schema rows : {count_extended}"
    )
    print(
        f"Total data rows      : {len(df)}"
    )

    check(
        count_original + count_extended == len(df),
        "all R9 rows were reconstructed",
    )

    return df


# =============================================================================
# SUMMARY
# =============================================================================

def summarize_group(df):
    energy_mean, energy_sd = mean_sd(
        df["energy_j_per_trial"]
    )

    wall_mean, wall_sd = mean_sd(
        df["wall_seconds"]
    )

    power_mean, power_sd = mean_sd(
        df["average_system_power_w"]
    )

    latency_mean, latency_sd = mean_sd(
        df["benchmark_mean_latency_ms"]
        if "benchmark_mean_latency_ms" in df.columns
        else df["mean_latency_ms"]
    )

    result = {
        "n_valid": int(len(df)),
        "energy_j_per_trial_mean": energy_mean,
        "energy_j_per_trial_sd": energy_sd,
        "wall_seconds_mean": wall_mean,
        "wall_seconds_sd": wall_sd,
        "average_power_w_mean": power_mean,
        "average_power_w_sd": power_sd,
        "mean_latency_ms_mean": latency_mean,
        "mean_latency_ms_sd": latency_sd,
    }

    if "pi_temp_start_c" in df.columns:
        x = pd.to_numeric(
            df["pi_temp_start_c"],
            errors="coerce",
        ).dropna()

        if len(x):
            result["temp_start_min_c"] = float(x.min())
            result["temp_start_max_c"] = float(x.max())

    if "pi_temp_end_c" in df.columns:
        x = pd.to_numeric(
            df["pi_temp_end_c"],
            errors="coerce",
        ).dropna()

        if len(x):
            result["temp_end_min_c"] = float(x.min())
            result["temp_end_max_c"] = float(x.max())

    return result


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 112)
    print(
        "R21 — PI 5 ENERGY-MEASUREMENT "
        "REPRODUCIBILITY AND VALIDITY AUDIT"
    )
    print("=" * 112)

    # -------------------------------------------------------------------------
    # Raw artifacts
    # -------------------------------------------------------------------------

    check(
        R9_CSV.exists(),
        "R9 raw energy summary exists",
    )

    check(
        R14C_CSV.exists(),
        "R14-C raw energy summary exists",
    )

    r9_sha = sha256_file(R9_CSV)
    r14_sha = sha256_file(R14C_CSV)

    print()
    print("RAW ARTIFACTS")
    print("-" * 112)

    print(f"R9    : {R9_CSV}")
    print(f"SHA256: {r9_sha}")
    print()

    print(f"R14-C : {R14C_CSV}")
    print(f"SHA256: {r14_sha}")

    # -------------------------------------------------------------------------
    # Read
    # -------------------------------------------------------------------------

    r9 = load_r9_historical_csv(
        R9_CSV
    )

    r14 = pd.read_csv(
        R14C_CSV
    )

    check(
        len(r9) == 13,
        "R9 contains 13 recorded rows",
    )

    check(
        len(r14) == 3,
        "R14-C contains 3 recorded rows",
    )

    # -------------------------------------------------------------------------
    # Build audit table
    # -------------------------------------------------------------------------

    records = []

    for _, row in r9.iterrows():
        rec = row.to_dict()

        pipeline = str(
            row["pipeline"]
        )

        run_number = int(
            row["run_number"]
        )

        key = (
            pipeline,
            run_number,
        )

        if key in INVALID_R9:
            valid = False
            reason = INVALID_R9[key]
        else:
            valid = True
            reason = "VALID"

        rec["source_stage"] = "R9"
        rec[
            "valid_for_final_statistics"
        ] = valid
        rec[
            "validity_reason"
        ] = reason

        records.append(rec)

    for _, row in r14.iterrows():
        rec = row.to_dict()

        rec["source_stage"] = "R14-C"
        rec[
            "valid_for_final_statistics"
        ] = True
        rec[
            "validity_reason"
        ] = "VALID"

        records.append(rec)

    audit = pd.DataFrame(
        records
    )

    # -------------------------------------------------------------------------
    # Validate run selection
    # -------------------------------------------------------------------------

    print()
    print("RUN VALIDITY")
    print("-" * 112)

    for pipeline, expected in EXPECTED_VALID_R9.items():
        mask = (
            (audit["source_stage"] == "R9")
            & (audit["pipeline"] == pipeline)
            & (
                audit[
                    "valid_for_final_statistics"
                ]
                == True
            )
        )

        observed = set(
            audit.loc[
                mask,
                "run_number",
            ]
            .astype(int)
            .tolist()
        )

        check(
            observed == expected,
            f"R9 {pipeline} valid runs = "
            f"{sorted(expected)}",
        )

    for pipeline, expected in EXPECTED_VALID_R14C.items():
        mask = (
            (audit["source_stage"] == "R14-C")
            & (audit["pipeline"] == pipeline)
            & (
                audit[
                    "valid_for_final_statistics"
                ]
                == True
            )
        )

        observed = set(
            audit.loc[
                mask,
                "run_number",
            ]
            .astype(int)
            .tolist()
        )

        check(
            observed == expected,
            f"R14-C {pipeline} valid runs = "
            f"{sorted(expected)}",
        )

    invalid = audit[
        audit[
            "valid_for_final_statistics"
        ]
        == False
    ]

    check(
        len(invalid) == 1,
        "exactly one historical measurement "
        "is invalidated",
    )

    bad = invalid.iloc[0]

    check(
        (
            bad["source_stage"] == "R9"
            and bad["pipeline"] == "STATIC_B0"
            and int(bad["run_number"]) == 2
        ),
        "only R9 STATIC_B0 run 2 is invalid",
    )

    print()
    print(
        "Invalidated historical measurement:"
    )
    print(
        "  R9 / STATIC_B0 / run 2"
    )
    print(
        "  reason: "
        + str(
            bad["validity_reason"]
        )
    )
    print(
        "  replacement: R9 / STATIC_B0 / run 4"
    )

    # -------------------------------------------------------------------------
    # Valid rows
    # -------------------------------------------------------------------------

    valid = audit[
        audit[
            "valid_for_final_statistics"
        ]
        == True
    ].copy()

    # Normalize R14-C latency column to the R9 name
    if "mean_latency_ms" in valid.columns:
        mask_r14 = (
            valid["source_stage"]
            == "R14-C"
        )

        valid.loc[
            mask_r14,
            "benchmark_mean_latency_ms",
        ] = valid.loc[
            mask_r14,
            "mean_latency_ms",
        ]

    # -------------------------------------------------------------------------
    # Valid-only summaries
    # -------------------------------------------------------------------------

    print()
    print("VALID-ONLY ENERGY SUMMARY")
    print("-" * 112)

    order = [
        ("R9", "STATIC_B0"),
        ("R14-C", "STATIC_B2"),
        ("R9", "STATIC_B6"),
        ("R9", "SEQUENTIAL_ALL"),
        ("R9", "FROZEN_B10"),
    ]

    summary_rows = []
    summaries = {}

    for source_stage, pipeline in order:
        d = valid[
            (
                valid["source_stage"]
                == source_stage
            )
            & (
                valid["pipeline"]
                == pipeline
            )
        ].copy()

        check(
            len(d) == 3,
            f"{source_stage} {pipeline} "
            "has 3 valid repetitions",
        )

        summary = summarize_group(
            d
        )

        key = (
            f"{source_stage}:"
            f"{pipeline}"
        )

        summaries[key] = summary

        summary_rows.append(
            {
                "source_stage": source_stage,
                "pipeline": pipeline,
                **summary,
            }
        )

        print(
            f"{pipeline:15s} | "
            f"n={summary['n_valid']} | "
            f"E="
            f"{summary['energy_j_per_trial_mean']:.6f} "
            f"+/- "
            f"{summary['energy_j_per_trial_sd']:.6f} "
            f"J/trial | "
            f"P="
            f"{summary['average_power_w_mean']:.6f} "
            f"+/- "
            f"{summary['average_power_w_sd']:.6f} W | "
            f"T="
            f"{summary['wall_seconds_mean']:.3f} "
            f"+/- "
            f"{summary['wall_seconds_sd']:.3f} s"
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    # -------------------------------------------------------------------------
    # Temperature evidence
    # -------------------------------------------------------------------------

    print()
    print("TEMPERATURE EVIDENCE")
    print("-" * 112)

    r9_valid = valid[
        valid["source_stage"]
        == "R9"
    ].copy()

    r9_temp_start = pd.to_numeric(
        r9_valid["pi_temp_start_c"],
        errors="coerce",
    ).dropna()

    r9_temp_end = pd.to_numeric(
        r9_valid["pi_temp_end_c"],
        errors="coerce",
    ).dropna()

    if len(r9_temp_start):
        print(
            "R9 valid start temperature range: "
            f"{r9_temp_start.min():.2f}"
            "–"
            f"{r9_temp_start.max():.2f} C"
        )
    else:
        print(
            "R9 valid start temperature: "
            "not recorded for all historical runs"
        )

    if len(r9_temp_end):
        print(
            "R9 valid end temperature range  : "
            f"{r9_temp_end.min():.2f}"
            "–"
            f"{r9_temp_end.max():.2f} C"
        )
    else:
        print(
            "R9 valid end temperature: "
            "not recorded for all historical runs"
        )

    r14_valid = valid[
        valid["source_stage"]
        == "R14-C"
    ].copy()

    r14_temp_start = pd.to_numeric(
        r14_valid["pi_temp_start_c"],
        errors="coerce",
    ).dropna()

    r14_temp_end = pd.to_numeric(
        r14_valid["pi_temp_end_c"],
        errors="coerce",
    ).dropna()

    print(
        "R14-C b2 start temperature range: "
        f"{r14_temp_start.min():.2f}"
        "–"
        f"{r14_temp_start.max():.2f} C"
    )

    print(
        "R14-C b2 end temperature range  : "
        f"{r14_temp_end.min():.2f}"
        "–"
        f"{r14_temp_end.max():.2f} C"
    )

    # -------------------------------------------------------------------------
    # Measurement precision
    # -------------------------------------------------------------------------

    print()
    print("MEASUREMENT-PRECISION AUDIT")
    print("-" * 112)

    print(
        "R9 preserved cumulative-energy readings "
        "are recorded to two decimal places in the CSV."
    )

    print(
        "R14-C preserved cumulative-energy readings "
        "are recorded to one decimal place in the CSV."
    )

    print(
        "Recorded decimal precision is NOT interpreted "
        "as the AT085 hardware accuracy or resolution."
    )

    print(
        "R14-C STATIC_B2 energy must therefore be "
        "reported approximately (≈5.76 J/trial), "
        "not as high-precision 5.760 +/- 0.000 J/trial."
    )

    print(
        "No historical throttling-status field is present "
        "in the preserved energy CSV artifacts."
    )

    print(
        "Therefore absence of thermal throttling is NOT "
        "claimed from these artifacts alone."
    )

    # -------------------------------------------------------------------------
    # B10 energy savings
    # -------------------------------------------------------------------------

    b6_energy = summaries[
        "R9:STATIC_B6"
    ][
        "energy_j_per_trial_mean"
    ]

    seq_energy = summaries[
        "R9:SEQUENTIAL_ALL"
    ][
        "energy_j_per_trial_mean"
    ]

    b10_energy = summaries[
        "R9:FROZEN_B10"
    ][
        "energy_j_per_trial_mean"
    ]

    saving_vs_b6 = (
        100.0
        * (
            b6_energy
            - b10_energy
        )
        / b6_energy
    )

    saving_vs_seq = (
        100.0
        * (
            seq_energy
            - b10_energy
        )
        / seq_energy
    )

    print()
    print("VALID-ONLY B10 ENERGY SAVINGS")
    print("-" * 112)

    print(
        "vs Static-b6      : "
        f"{saving_vs_b6:.6f}%"
    )

    print(
        "vs Sequential-All : "
        f"{saving_vs_seq:.6f}%"
    )

    # -------------------------------------------------------------------------
    # Save audit artifacts
    # -------------------------------------------------------------------------

    audit.to_csv(
        OUT_RUN_AUDIT,
        index=False,
    )

    summary_df.to_csv(
        OUT_SUMMARY,
        index=False,
    )

    report = {
        "stage": "R21",
        "status": "PASS",
        "raw_artifacts": {
            "r9_csv": str(R9_CSV),
            "r9_sha256": r9_sha,
            "r14c_csv": str(R14C_CSV),
            "r14c_sha256": r14_sha,
        },
        "invalid_measurements": [
            {
                "source_stage": "R9",
                "pipeline": "STATIC_B0",
                "run_number": 2,
                "replacement_run": 4,
                "reason": INVALID_R9[
                    (
                        "STATIC_B0",
                        2,
                    )
                ],
            }
        ],
        "valid_run_policy": {
            "R9": {
                pipeline: sorted(
                    runs
                )
                for pipeline, runs
                in EXPECTED_VALID_R9.items()
            },
            "R14-C": {
                pipeline: sorted(
                    runs
                )
                for pipeline, runs
                in EXPECTED_VALID_R14C.items()
            },
        },
        "valid_only_summary": summaries,
        "b10_energy_savings_percent": {
            "vs_static_b6": (
                saving_vs_b6
            ),
            "vs_sequential_all": (
                saving_vs_seq
            ),
        },
        "measurement_notes": {
            "meter": (
                "AT085 Type-C Tester"
            ),
            "measurement_location": (
                "Raspberry Pi 5 power input"
            ),
            "energy_definition": (
                "Difference between cumulative Wh readings "
                "at the beginning and end of the timed "
                "500-trial benchmark interval."
            ),
            "r9_recorded_wh_decimal_places": 2,
            "r14c_recorded_wh_decimal_places": 1,
            "hardware_accuracy_or_resolution": (
                "Not inferred from CSV decimal precision."
            ),
            "throttling_status": (
                "Not present in preserved energy CSV artifacts."
            ),
            "static_b0_run2": (
                "Protocol-invalid measurement. End cumulative "
                "energy was not recorded at the prescribed "
                "benchmark boundary. Run 4 was deliberately "
                "performed as its replacement."
            ),
        },
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
            ensure_ascii=False,
        )

    # -------------------------------------------------------------------------
    # Final
    # -------------------------------------------------------------------------

    print()
    print("OUTPUT")
    print("-" * 112)

    print(
        f"Run audit : {OUT_RUN_AUDIT}"
    )

    print(
        f"Summary   : {OUT_SUMMARY}"
    )

    print(
        f"JSON      : {OUT_JSON}"
    )

    print()
    print(
        "PASS — energy repeatability and measurement-validity "
        "audit completed without modifying historical raw data."
    )

    print("=" * 112)


if __name__ == "__main__":
    main()