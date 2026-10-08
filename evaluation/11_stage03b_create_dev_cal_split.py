import json
import random
import hashlib
from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT
# ============================================================

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]


# ============================================================
# INPUT / OUTPUT
# ============================================================

AUDIT_CSV = (
    PROJECT_ROOT
    / "results"
    / "stage03_dev_cal"
    / "stage03a_dev_pool_audit"
    / "vox1_dev_speaker_summary.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "stage03_dev_cal"
    / "stage03b_dev_cal_split"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_CSV = OUTPUT_DIR / "dev_cal_speaker_manifest.csv"
DEV_CSV = OUTPUT_DIR / "dev_speakers.csv"
CAL_CSV = OUTPUT_DIR / "cal_speakers.csv"
JSON_OUT = OUTPUT_DIR / "dev_cal_split.json"
FINGERPRINT_TXT = OUTPUT_DIR / "dev_cal_split_sha256.txt"


# ============================================================
# CONFIG
# ============================================================

SEED = 42
DEV_RATIO = 0.80

EXPECTED_TOTAL_SPEAKERS = 1211
EXPECTED_DEV_SPEAKERS = 969
EXPECTED_CAL_SPEAKERS = 242


# ============================================================
# HELPERS
# ============================================================

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_manifest_bytes(df: pd.DataFrame) -> bytes:
    canonical = (
        df[
            ["speaker_id", "split", "n_utterances"]
        ]
        .sort_values("speaker_id")
        .to_csv(index=False, lineterminator="\n")
    )
    return canonical.encode("utf-8")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 92)
    print("STAGE 03B — DETERMINISTIC SPEAKER-DISJOINT DEV/CAL SPLIT")
    print("=" * 92)

    if not AUDIT_CSV.exists():
        raise FileNotFoundError(AUDIT_CSV)

    df = pd.read_csv(AUDIT_CSV)

    required = {
        "speaker_id",
        "n_utterances",
        "eligible_dev",
    }

    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Missing required columns: {sorted(missing)}"
        )

    eligible = (
        df[df["eligible_dev"] == True]
        [["speaker_id", "n_utterances"]]
        .copy()
    )

    if len(eligible) != EXPECTED_TOTAL_SPEAKERS:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_SPEAKERS} eligible speakers, "
            f"found {len(eligible)}"
        )

    if eligible["speaker_id"].duplicated().any():
        raise RuntimeError("Duplicate speaker_id detected.")

    # Deterministic speaker-level split.
    speakers = sorted(
        eligible["speaker_id"].astype(str).tolist()
    )

    rng = random.Random(SEED)
    rng.shuffle(speakers)

    n_dev = round(len(speakers) * DEV_RATIO)

    # Enforce the LOCKED 969 / 242 split.
    if n_dev != EXPECTED_DEV_SPEAKERS:
        raise RuntimeError(
            f"Unexpected DEV count from ratio: {n_dev}"
        )

    dev_speakers = set(speakers[:n_dev])
    cal_speakers = set(speakers[n_dev:])

    if len(dev_speakers) != EXPECTED_DEV_SPEAKERS:
        raise RuntimeError("DEV speaker count mismatch.")

    if len(cal_speakers) != EXPECTED_CAL_SPEAKERS:
        raise RuntimeError("CAL speaker count mismatch.")

    if dev_speakers & cal_speakers:
        raise RuntimeError("DEV/CAL overlap detected.")

    eligible["split"] = eligible["speaker_id"].map(
        lambda x: "DEV" if x in dev_speakers else "CAL"
    )

    manifest = (
        eligible[
            ["speaker_id", "split", "n_utterances"]
        ]
        .sort_values(
            ["split", "speaker_id"]
        )
        .reset_index(drop=True)
    )

    dev_df = (
        manifest[manifest["split"] == "DEV"]
        .sort_values("speaker_id")
        .reset_index(drop=True)
    )

    cal_df = (
        manifest[manifest["split"] == "CAL"]
        .sort_values("speaker_id")
        .reset_index(drop=True)
    )

    dev_audio = int(dev_df["n_utterances"].sum())
    cal_audio = int(cal_df["n_utterances"].sum())

    manifest.to_csv(
        MANIFEST_CSV,
        index=False,
    )

    dev_df.to_csv(
        DEV_CSV,
        index=False,
    )

    cal_df.to_csv(
        CAL_CSV,
        index=False,
    )

    fingerprint = sha256_bytes(
        canonical_manifest_bytes(manifest)
    )

    FINGERPRINT_TXT.write_text(
        fingerprint + "\n",
        encoding="utf-8",
    )

    report = {
        "stage": "03B",
        "seed": SEED,
        "dev_ratio": DEV_RATIO,
        "total_speakers": len(manifest),
        "dev_speakers": len(dev_df),
        "cal_speakers": len(cal_df),
        "dev_audio_files": dev_audio,
        "cal_audio_files": cal_audio,
        "speaker_overlap_dev_cal": 0,
        "sha256_manifest": fingerprint,
        "split_policy": (
            "speaker-level deterministic split: sorted speaker IDs, "
            "Python random.Random(seed=42) shuffle, first 969 to DEV, "
            "remaining 242 to CAL"
        ),
    }

    with open(
        JSON_OUT,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Total speakers : {len(manifest)}")
    print(f"DEV speakers   : {len(dev_df)}")
    print(f"CAL speakers   : {len(cal_df)}")
    print(f"DEV/CAL overlap: {len(dev_speakers & cal_speakers)}")
    print()
    print(f"DEV audio files: {dev_audio}")
    print(f"CAL audio files: {cal_audio}")
    print()
    print(f"SHA256         : {fingerprint}")
    print()
    print("Outputs:")
    print(MANIFEST_CSV)
    print(DEV_CSV)
    print(CAL_CSV)
    print(JSON_OUT)
    print(FINGERPRINT_TXT)
    print()
    print("PASS — deterministic speaker-disjoint DEV/CAL split created.")
    print("=" * 92)


if __name__ == "__main__":
    main()
