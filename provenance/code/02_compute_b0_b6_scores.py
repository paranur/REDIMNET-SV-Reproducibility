import sys
import csv
from pathlib import Path

import numpy as np
from tqdm import tqdm

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
sys.path.append(str(PROJECT_ROOT))

from configs.config import (
    CANONICAL_TRIALS,
    VOX1O_ROOT,
    SMALL_MODEL,
    LARGE_MODEL,
    B0_EMB_DIR,
    B6_EMB_DIR,
    SCORES_CSV,
)


def safe_name(rel_path: str):
    rel = Path(rel_path).with_suffix("")
    return "__".join(rel.parts) + ".npy"


def load_embedding(folder: Path, rel_path: str):
    path = folder / safe_name(rel_path)
    if not path.exists():
        raise FileNotFoundError(path)

    emb = np.load(path)
    if emb.ndim == 2 and emb.shape[0] == 1:
        emb = emb[0]
    if emb.ndim != 1:
        raise ValueError(f"Unexpected embedding shape {emb.shape}: {path}")

    emb = emb.astype(np.float32)
    emb /= np.linalg.norm(emb) + 1e-12
    return emb


def parse_trials():
    rows = []
    with open(CANONICAL_TRIALS, "r", encoding="utf-8") as f:
        for trial_id, line in enumerate(f):
            s = line.strip()
            if not s:
                continue
            label, rel1, rel2 = s.split()
            rows.append((trial_id, int(label), rel1, rel2))
    return rows


def cosine(a, b):
    return float(np.dot(a, b))


def main():
    trials = parse_trials()
    SCORES_CSV.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "trial_id",
        "label",
        "enroll_path",
        "test_path",
        "score_b0",
        "score_b6",
        "small_model",
        "large_model",
        "dataset",
    ]

    failed = 0

    with open(SCORES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for trial_id, label, rel1, rel2 in tqdm(trials, desc="Paired b0/b6 scoring"):
            try:
                b0_a = load_embedding(B0_EMB_DIR, rel1)
                b0_b = load_embedding(B0_EMB_DIR, rel2)
                b6_a = load_embedding(B6_EMB_DIR, rel1)
                b6_b = load_embedding(B6_EMB_DIR, rel2)

                s0 = cosine(b0_a, b0_b)
                s6 = cosine(b6_a, b6_b)

                writer.writerow({
                    "trial_id": trial_id,
                    "label": label,
                    "enroll_path": rel1,
                    "test_path": rel2,
                    "score_b0": f"{s0:.8f}",
                    "score_b6": f"{s6:.8f}",
                    "small_model": SMALL_MODEL,
                    "large_model": LARGE_MODEL,
                    "dataset": "VoxCeleb1-O",
                })
            except Exception as exc:
                failed += 1
                print(
                    f"\n[FAILED] trial={trial_id} "
                    f"enroll={rel1} test={rel2} error={repr(exc)}"
                )

    if failed:
        raise RuntimeError(f"{failed} trials failed during scoring.")

    print("\nPASS — paired score table created.")
    print(f"Trials: {len(trials)}")
    print(f"Saved: {SCORES_CSV}")


if __name__ == "__main__":
    main()
