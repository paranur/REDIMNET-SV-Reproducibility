# REDIMNET-SV score-level reproducibility

Margin-Guided Adaptive Model Routing for Resource-Efficient Speaker Verification on Edge Devices. Revision evidence for MDPI Computation, computation-4621309, Round 2.

This package recomputes reported numerical analyses from frozen released scores. It does not provide end-to-end audio/model reproduction. It contains no recordings, embeddings, checkpoint binaries or private datasets. The original public release is retained in Git history; its exact commit and file inventory appear in INITIAL_REPOSITORY_AUDIT.md.

## Structure

- `protocols/`: DEV/CAL speaker and trial manifests, official original/cleaned VoxCeleb1-O verification lists, USC-SV identifiers, protocol-independence evidence and Pi trial manifest.
- `scores/`: original PTN, DEV/CAL and USC score tables; `ftlm/` contains FT-LM DEV/CAL/test scores and frozen routing audit; `cleaned/` contains the existing cleaned-protocol scores and audit.
- `policies/`: CAL-frozen PTN/FT-LM decision thresholds, DEV-frozen routing policies and R2B mixed-decision threshold. Values are released unchanged.
- `evaluation/`: portable reproduction commands listed below, plus original archival stage scripts.
- `evidence/mr01/`, `evidence/mr02/`: previously executed expected CSV tables, per-trial labels/transitions, replicate results, speaker multiplicities and pre-execution bootstrap design.
- `evidence/reference/`: reported primary, FT-LM, cleaned and USC numerical references; `evidence/historical/`: retained R5/R13/R15/R26 evidence used for provenance or comparisons.
- `deployment/`: previously released Pi latency, energy, memory and lazy-enrollment evidence. Hardware scripts are archival and are not executed by score reproduction.
- `provenance/`: original/released SHA-256 mappings, metadata-only normalizations, checkpoint identifiers/hashes, original public file inventory, and sanitized provenance code.
- `RELEASE_MANIFEST.csv`: complete file-level inventory with hashes, sizes and added/updated/unchanged status (excludes itself).
- `SHA256SUMS.txt`: hashes excluding itself and RELEASE_MANIFEST.csv to avoid circular self-hashing.
- `reproduced/`: newly generated outputs; ignored by Git and never substituted for frozen evidence.

## Environment and installation

Tested with existing CPython 3.10.0, NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.7.2, joblib 1.5.3 and threadpoolctl 3.6.0. MR-01 uses only the standard library; remaining numerical checks require the pinned packages. No pandas, Torch, GPU, audio access or network connection is needed for reproduction.

For a researcher setting up their own environment, install dependencies with `python -m pip install -r requirements.txt`. The release preparation installed no packages and created no environments. Use the tested versions for exact bootstrap replication; other versions have not been validated.

Clone this existing repository, enter its root, then run:

```sh
git clone https://github.com/paranur/REDIMNET-SV-Reproducibility.git
cd REDIMNET-SV-Reproducibility
python evaluation/verify_release_integrity.py
python evaluation/run_package_reproduction.py
```

The reproduction runner executes all three portable analyses under a file-access guard that rejects external experimental files and network access. Interpreter/dependency files within the installed Python prefix are allowed. Every scientific source comes from this checkout. It compares all 14 expected MR-01/MR-02 CSV tables (including replicate-level results) and checks 21 existing score-level metric rows. All results must report PASS. Runtime logs, access audits and the aggregate report are written under `reproduced/`.

Individual commands, from repository root:

```sh
python evaluation/evaluate_released_scores.py
python evaluation/recalculate_independent_expert_benefit.py
python evaluation/corrected_speaker_dependent_paired_bootstrap.py
```

Use `python evaluation/enforce_package_inputs.py evaluation/<script>.py` to apply the same file/network guard individually. Do not run the numbered archival stage or deployment scripts as score-level reproduction: some describe original trial construction, threshold development or hardware inference and require excluded resources. The portable evaluator verifies existing frozen artifacts; it never reruns those development stages.

## Expected numerical outputs

| Original VoxCeleb1-O system | EER (%) | Normalized minDCF |
| --- | ---: | ---: |
| Static-b0 | 1.540296925 | 0.163997879 |
| Static-b6 | 0.715800636 | 0.054347826 |
| Frozen-B10 | 0.723753977 | 0.135206787 |

MR-01: 37,720 trials; 396 potential benefits; 100 potential harms; 683 b0 errors; prevalence 0.010498409332; ROC-AUC 0.979998381629; average precision 0.298318557271; AP/prevalence lift 28.415595910. Benefit recalls in the lowest margins 5/10/20/30%: 351/396, 382/396, 394/396 and 395/396. Actual B10: 3,641 routed, 378 corrections, 102 harms, 276 net corrections. Potential labels use separate CAL-frozen b0/b6 decisions on all trials and do not depend on routing. Actual B10 transitions use the separate CAL-frozen mixed threshold. PR-AUC primary estimator is grouped average precision; trapezoidal PR area is additionally reported. Margin ties are broken deterministically by trial ID for region membership.

MR-02: 2,000 paired speaker draws, NumPy PCG64 seed 20261020. Each target weight is M_s exactly once; distinct-speaker impostor weight is M_s*M_t; the same weights apply to all three systems. EER uses the first canonical ROC point minimizing absolute FNR-FPR, with score ties grouped and default ROC pruning; no interpolation. Normalized minDCF uses P_target=0.01 and unit costs. CI endpoints use NumPy linear 2.5/97.5 percentiles. See evidence/mr02/bootstrap_design.md for the mathematical construction and limitations.

| Paired difference | EER 95% CI (pp) | minDCF 95% CI |
| --- | --- | --- |
| B10 minus b0 | [-1.136840702, -0.491999359] | [-0.050345451, -0.016909928] |
| B10 minus b6 | [-0.009712816, 0.142034083] | [0.010563772, 0.120433435] |
| b6 minus b0 | [-1.208331421, -0.510352215] | [-0.148207304, -0.045522239] |

An interval containing zero does not establish equivalence or noninferiority. These are approximate pointwise percentile intervals conditional on the fixed 40-speaker protocol.

FT-LM EER-B10: 3,773 routed, EER 0.556733828%, minDCF 0.075291622. Cleaned VoxCeleb1-O: 37,611 trials, 18,802 target and 18,809 impostor; Frozen-B10 routes 3,604 trials, EER 0.571641790%, minDCF 0.132740077. All three FT-LM static systems and all six frozen FT-LM policies are checked. USC CAL/TEST metrics and TEST frozen-operating-point FA/MISS counts are checked against archived references. DEV/CAL PTN and FT-LM trial identities, speaker counts (969/242) and disjointness are verified.

## Integrity, provenance and exclusions

Source hashes record exact source identities, including original scientific bytes and the release-export bytes where machine-root metadata was normalized. Score values, labels, trial pairings, thresholds and routing bits are not changed. The initial Git release normalized line endings while its checksum list referred to original CRLF bytes. This update preserves exact original scientific-source bytes using .gitattributes; provenance/baseline_compatibility.csv checks every old public file for metadata-only or newline-only differences. See provenance/public_metadata_normalizations.csv for existing public files with path-only changes, and provenance/source_artifact_manifest.csv for copied-source hashes. The original source manifests' absolute machine paths are not published.

Checkpoint names and byte hashes are provided for PTN and FT-LM b0/b2/b6. Audio-level inference, embedding generation, original training, checkpoint conversion, renewed hardware measurements and private upstream USC source-manifest reconstruction are NOT REPRODUCIBLE from this package alone. These limitations do not prevent the released score-level tests. No new scientific experiments, threshold fitting or policy selection are performed.

See LICENSE_NOTES.md for dataset attribution, upstream terms, the absence of a new repository-wide license grant, and reuse limitations. Cite the associated manuscript plus VoxCeleb, USC and ReDimNet sources as appropriate. Publication metadata will be updated when available. See VALIDATION_REPORT.md and RELEASE_NOTES.md for the executed release checks and exact file changes.
