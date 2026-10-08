import json
import random
import hashlib
from pathlib import Path
from collections import defaultdict

import pandas as pd


# ============================================================
# PROJECT
# ============================================================

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]


# ============================================================
# INPUTS
# ============================================================

VOX1_ROOT = Path("DATASETS/VoxCeleb1/wav")

SPLIT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "stage03_dev_cal"
    / "stage03b_dev_cal_split"
)

DEV_SPEAKERS_CSV = SPLIT_ROOT / "dev_speakers.csv"
CAL_SPEAKERS_CSV = SPLIT_ROOT / "cal_speakers.csv"
SPLIT_SHA256_TXT = SPLIT_ROOT / "dev_cal_split_sha256.txt"


# ============================================================
# OUTPUTS
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "stage03_dev_cal"
    / "stage03c_trials"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEV_TRIALS_CSV = OUTPUT_DIR / "dev_trials_50k.csv"
CAL_TRIALS_CSV = OUTPUT_DIR / "cal_trials_20k.csv"

DEV_TRIALS_TXT = OUTPUT_DIR / "dev_trials_50k.txt"
CAL_TRIALS_TXT = OUTPUT_DIR / "cal_trials_20k.txt"

DEV_SHA256_TXT = OUTPUT_DIR / "dev_trials_50k_sha256.txt"
CAL_SHA256_TXT = OUTPUT_DIR / "cal_trials_20k_sha256.txt"

REPORT_JSON = OUTPUT_DIR / "dev_cal_trials_report.json"


# ============================================================
# CONFIG
# ============================================================

SEED = 42

DEV_TARGET = 25_000
DEV_IMPOSTOR = 25_000

CAL_TARGET = 10_000
CAL_IMPOSTOR = 10_000

AUDIO_EXTS = {".wav"}

# Prefer target pairs from different VoxCeleb video/session folders.
PREFER_CROSS_SESSION_TARGETS = True

MAX_ATTEMPTS_MULTIPLIER = 200


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def read_locked_split_hash() -> str:
    if not SPLIT_SHA256_TXT.exists():
        raise FileNotFoundError(SPLIT_SHA256_TXT)

    value = SPLIT_SHA256_TXT.read_text(
        encoding="utf-8"
    ).strip()

    if len(value) != 64:
        raise RuntimeError(
            f"Invalid DEV/CAL split SHA256: {value}"
        )

    return value


def load_speaker_ids(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    if "speaker_id" not in df.columns:
        raise RuntimeError(
            f"{path} has no speaker_id column"
        )

    ids = sorted(
        df["speaker_id"].astype(str).tolist()
    )

    if len(ids) != len(set(ids)):
        raise RuntimeError(
            f"Duplicate speaker IDs in {path}"
        )

    return ids


def build_speaker_audio_index(speaker_ids):
    index = {}

    for i, spk in enumerate(speaker_ids, start=1):
        spk_dir = VOX1_ROOT / spk

        if not spk_dir.exists():
            raise FileNotFoundError(spk_dir)

        files = sorted(
            p
            for p in spk_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() in AUDIO_EXTS
        )

        if len(files) < 2:
            raise RuntimeError(
                f"Speaker {spk} has fewer than 2 audio files"
            )

        rels = [
            p.relative_to(VOX1_ROOT).as_posix()
            for p in files
        ]

        index[spk] = rels

        if i % 100 == 0 or i == len(speaker_ids):
            print(
                f"Indexed {i}/{len(speaker_ids)} speakers"
            )

    return index


def session_of(rel_path: str) -> str:
    # Expected VoxCeleb1 layout:
    # speaker_id / video_or_session_id / file.wav
    parts = Path(rel_path).parts

    if len(parts) >= 3:
        return parts[1]

    return ""


def unordered_pair_key(path1: str, path2: str):
    return tuple(
        sorted((path1, path2))
    )


def generate_target_trials(
    speaker_audio,
    n_trials,
    rng,
):
    speakers = sorted(speaker_audio.keys())

    trials = []
    seen = set()

    cross_session_count = 0
    same_session_count = 0

    max_attempts = n_trials * MAX_ATTEMPTS_MULTIPLIER
    attempts = 0

    while len(trials) < n_trials:
        attempts += 1

        if attempts > max_attempts:
            raise RuntimeError(
                f"Could not generate {n_trials} unique target trials. "
                f"Generated {len(trials)}."
            )

        spk = rng.choice(speakers)
        audios = speaker_audio[spk]

        if len(audios) < 2:
            continue

        a, b = rng.sample(audios, 2)

        if a == b:
            continue

        if PREFER_CROSS_SESSION_TARGETS:
            s1 = session_of(a)
            s2 = session_of(b)

            if s1 == s2:
                # Try to replace b with an utterance from another session.
                alternatives = [
                    x
                    for x in audios
                    if x != a
                    and session_of(x) != s1
                ]

                if alternatives:
                    b = rng.choice(alternatives)

        key = unordered_pair_key(a, b)

        if key in seen:
            continue

        seen.add(key)

        if session_of(a) != session_of(b):
            cross_session_count += 1
        else:
            same_session_count += 1

        trials.append(
            {
                "label": 1,
                "enroll_path": a,
                "test_path": b,
                "enroll_speaker": spk,
                "test_speaker": spk,
                "cross_session": (
                    session_of(a) != session_of(b)
                ),
            }
        )

    return (
        trials,
        {
            "cross_session_target_trials":
                cross_session_count,
            "same_session_target_trials":
                same_session_count,
        },
    )


def generate_impostor_trials(
    speaker_audio,
    n_trials,
    rng,
):
    speakers = sorted(speaker_audio.keys())

    trials = []
    seen = set()

    max_attempts = n_trials * MAX_ATTEMPTS_MULTIPLIER
    attempts = 0

    while len(trials) < n_trials:
        attempts += 1

        if attempts > max_attempts:
            raise RuntimeError(
                f"Could not generate {n_trials} unique impostor trials. "
                f"Generated {len(trials)}."
            )

        spk1, spk2 = rng.sample(
            speakers,
            2,
        )

        a = rng.choice(
            speaker_audio[spk1]
        )

        b = rng.choice(
            speaker_audio[spk2]
        )

        key = unordered_pair_key(a, b)

        if key in seen:
            continue

        seen.add(key)

        trials.append(
            {
                "label": 0,
                "enroll_path": a,
                "test_path": b,
                "enroll_speaker": spk1,
                "test_speaker": spk2,
                "cross_session": True,
            }
        )

    return trials


def validate_trials(
    df,
    expected_target,
    expected_impostor,
    allowed_speakers,
    split_name,
):
    if len(df) != expected_target + expected_impostor:
        raise RuntimeError(
            f"{split_name}: trial count mismatch"
        )

    n_target = int(
        (df["label"] == 1).sum()
    )

    n_impostor = int(
        (df["label"] == 0).sum()
    )

    if n_target != expected_target:
        raise RuntimeError(
            f"{split_name}: target count mismatch "
            f"{n_target} != {expected_target}"
        )

    if n_impostor != expected_impostor:
        raise RuntimeError(
            f"{split_name}: impostor count mismatch "
            f"{n_impostor} != {expected_impostor}"
        )

    # Target must have same speaker.
    bad_target = df[
        (df["label"] == 1)
        & (
            df["enroll_speaker"]
            != df["test_speaker"]
        )
    ]

    if len(bad_target):
        raise RuntimeError(
            f"{split_name}: invalid target trials found"
        )

    # Impostor must have different speakers.
    bad_impostor = df[
        (df["label"] == 0)
        & (
            df["enroll_speaker"]
            == df["test_speaker"]
        )
    ]

    if len(bad_impostor):
        raise RuntimeError(
            f"{split_name}: invalid impostor trials found"
        )

    if (
        df["enroll_speaker"]
        .isin(allowed_speakers)
        .all()
        is False
    ):
        raise RuntimeError(
            f"{split_name}: enroll speaker outside split"
        )

    if (
        df["test_speaker"]
        .isin(allowed_speakers)
        .all()
        is False
    ):
        raise RuntimeError(
            f"{split_name}: test speaker outside split"
        )

    # No identical enrollment/test file.
    same_file = (
        df["enroll_path"]
        == df["test_path"]
    )

    if same_file.any():
        raise RuntimeError(
            f"{split_name}: identical enroll/test audio detected"
        )

    # No duplicate unordered audio pair.
    pair_keys = [
        unordered_pair_key(a, b)
        for a, b in zip(
            df["enroll_path"],
            df["test_path"],
        )
    ]

    if len(pair_keys) != len(set(pair_keys)):
        raise RuntimeError(
            f"{split_name}: duplicate unordered pairs detected"
        )


def save_trial_files(
    df,
    csv_path,
    txt_path,
):
    out = df.copy()
    out.insert(
        0,
        "trial_id",
        range(len(out)),
    )

    out.to_csv(
        csv_path,
        index=False,
    )

    with open(
        txt_path,
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:

        for row in out.itertuples(
            index=False
        ):
            f.write(
                f"{int(row.label)} "
                f"{row.enroll_path} "
                f"{row.test_path}\n"
            )

    return out


def build_trials_for_split(
    split_name,
    speaker_ids,
    n_target,
    n_impostor,
    seed_offset,
):
    print()
    print("=" * 92)
    print(
        f"BUILDING {split_name} TRIALS"
    )
    print("=" * 92)

    speaker_audio = build_speaker_audio_index(
        speaker_ids
    )

    rng = random.Random(
        SEED + seed_offset
    )

    target_trials, target_stats = (
        generate_target_trials(
            speaker_audio,
            n_target,
            rng,
        )
    )

    impostor_trials = (
        generate_impostor_trials(
            speaker_audio,
            n_impostor,
            rng,
        )
    )

    trials = (
        target_trials
        + impostor_trials
    )

    rng.shuffle(
        trials
    )

    df = pd.DataFrame(
        trials
    )

    validate_trials(
        df,
        n_target,
        n_impostor,
        set(speaker_ids),
        split_name,
    )

    return (
        df,
        speaker_audio,
        target_stats,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 92)
    print(
        "STAGE 03C — CANONICAL DEV/CAL VERIFICATION TRIAL GENERATION"
    )
    print("=" * 92)

    locked_split_sha256 = (
        read_locked_split_hash()
    )

    print(
        f"Locked DEV/CAL split SHA256: "
        f"{locked_split_sha256}"
    )

    dev_speakers = load_speaker_ids(
        DEV_SPEAKERS_CSV
    )

    cal_speakers = load_speaker_ids(
        CAL_SPEAKERS_CSV
    )

    if set(dev_speakers) & set(cal_speakers):
        raise RuntimeError(
            "DEV/CAL speaker overlap detected before trial generation"
        )

    print(
        f"DEV speakers: {len(dev_speakers)}"
    )

    print(
        f"CAL speakers: {len(cal_speakers)}"
    )

    # Independent RNG streams, still reproducible from master seed.
    dev_df, dev_audio, dev_target_stats = (
        build_trials_for_split(
            "DEV",
            dev_speakers,
            DEV_TARGET,
            DEV_IMPOSTOR,
            seed_offset=1000,
        )
    )

    cal_df, cal_audio, cal_target_stats = (
        build_trials_for_split(
            "CAL",
            cal_speakers,
            CAL_TARGET,
            CAL_IMPOSTOR,
            seed_offset=2000,
        )
    )

    # Strong separation audit.
    dev_trial_speakers = (
        set(dev_df["enroll_speaker"])
        | set(dev_df["test_speaker"])
    )

    cal_trial_speakers = (
        set(cal_df["enroll_speaker"])
        | set(cal_df["test_speaker"])
    )

    if dev_trial_speakers & cal_trial_speakers:
        raise RuntimeError(
            "Speaker overlap detected between DEV and CAL trials"
        )

    dev_saved = save_trial_files(
        dev_df,
        DEV_TRIALS_CSV,
        DEV_TRIALS_TXT,
    )

    cal_saved = save_trial_files(
        cal_df,
        CAL_TRIALS_CSV,
        CAL_TRIALS_TXT,
    )

    dev_hash = sha256_file(
        DEV_TRIALS_TXT
    )

    cal_hash = sha256_file(
        CAL_TRIALS_TXT
    )

    DEV_SHA256_TXT.write_text(
        dev_hash + "\n",
        encoding="utf-8",
    )

    CAL_SHA256_TXT.write_text(
        cal_hash + "\n",
        encoding="utf-8",
    )

    report = {
        "stage": "03C",
        "master_seed": SEED,
        "locked_dev_cal_split_sha256":
            locked_split_sha256,
        "trial_generation_policy": {
            "target": (
                "same speaker, different utterances; "
                "different session/video preferred when available"
            ),
            "impostor": (
                "different speakers"
            ),
            "duplicate_unordered_pairs":
                "forbidden within each split",
            "identical_enroll_test_audio":
                "forbidden",
            "shuffle":
                "deterministic",
        },
        "DEV": {
            "speakers": len(dev_speakers),
            "audio_files": sum(
                len(v)
                for v in dev_audio.values()
            ),
            "trials": len(dev_saved),
            "target": int(
                (dev_saved["label"] == 1).sum()
            ),
            "impostor": int(
                (dev_saved["label"] == 0).sum()
            ),
            **dev_target_stats,
            "trials_txt_sha256": dev_hash,
        },
        "CAL": {
            "speakers": len(cal_speakers),
            "audio_files": sum(
                len(v)
                for v in cal_audio.values()
            ),
            "trials": len(cal_saved),
            "target": int(
                (cal_saved["label"] == 1).sum()
            ),
            "impostor": int(
                (cal_saved["label"] == 0).sum()
            ),
            **cal_target_stats,
            "trials_txt_sha256": cal_hash,
        },
        "dev_cal_trial_speaker_overlap": 0,
    }

    with open(
        REPORT_JSON,
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
    print("=" * 92)
    print("FINAL AUDIT")
    print("=" * 92)

    print(
        f"DEV: {len(dev_saved)} trials | "
        f"target={(dev_saved['label'] == 1).sum()} | "
        f"impostor={(dev_saved['label'] == 0).sum()} | "
        f"cross-session target="
        f"{dev_target_stats['cross_session_target_trials']} | "
        f"same-session target="
        f"{dev_target_stats['same_session_target_trials']}"
    )

    print(
        f"CAL: {len(cal_saved)} trials | "
        f"target={(cal_saved['label'] == 1).sum()} | "
        f"impostor={(cal_saved['label'] == 0).sum()} | "
        f"cross-session target="
        f"{cal_target_stats['cross_session_target_trials']} | "
        f"same-session target="
        f"{cal_target_stats['same_session_target_trials']}"
    )

    print(
        f"DEV/CAL trial-speaker overlap: "
        f"{len(dev_trial_speakers & cal_trial_speakers)}"
    )

    print()
    print(
        f"DEV trials SHA256: {dev_hash}"
    )

    print(
        f"CAL trials SHA256: {cal_hash}"
    )

    print()
    print("Outputs:")
    print(DEV_TRIALS_CSV)
    print(CAL_TRIALS_CSV)
    print(DEV_TRIALS_TXT)
    print(CAL_TRIALS_TXT)
    print(REPORT_JSON)

    print()
    print(
        "PASS — canonical speaker-disjoint DEV/CAL verification trials created."
    )

    print("=" * 92)


if __name__ == "__main__":
    main()
