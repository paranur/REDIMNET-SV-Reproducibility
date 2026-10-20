# Initial public repository audit

Remote HEAD and local baseline: `adc1f366453bf4ef737c374c0a3a614e9300d59e`. All 51 tracked files inspected; full hashes in provenance/initial_public_file_inventory.csv. Existing local checkout clean and identical to remote HEAD.

| Category | Initial status | Evidence |
| --- | --- | --- |
| DEV/CAL speaker manifests | PRESENT | protocols/dev_cal_speaker_manifest.csv |
| verification trial manifests | PRESENT | protocols/dev_trials_50k.csv, protocols/cal_trials_20k.csv |
| frozen routing policies | PRESENT | policies/dev_margin_frozen_policies.json |
| CAL-derived thresholds | PRESENT | policies/cal_frozen_thresholds.json |
| VoxCeleb1-O per-trial scores | PRESENT | scores/vox1o_frozen_margin_router_trial_audit.csv |
| USC-SV CAL/TEST scores | PRESENT | scores/usc_cal_frozen_b10_scores.csv, scores/usc_test_final_frozen_scores.csv |
| FT-LM scores | MISSING | Full tracked tree checked; absent |
| cleaned VoxCeleb1-O scores | MISSING | Full tracked tree checked; absent |
| evaluation scripts | PRESENT | evaluation/22_stage04a_evaluate_frozen_router_vox1o.py |
| bootstrap scripts | MISSING | Full tracked tree checked; absent |
| independent expert-benefit scripts | MISSING | Full tracked tree checked; absent |
| deployment evidence | PRESENT | deployment/energy/r21_energy_reproducibility.json |
| source SHA-256 hashes | PRESENT | SHA256SUMS.txt |
| runnable package-relative execution instructions | MISSING | Full tracked tree checked; absent |

Existing evaluation files are archival stage scripts that expect the original results tree; the initial README does not provide working package-only numerical reproduction commands. Existing checksum file is present but lacks an executable verifier. Initial repository has no explicit repository-wide license.

Missing source protocol endpoints for primary VoxCeleb1-O prevented package-only speaker bootstrap; the new original-score table and official protocol supply them. No audio, embeddings, private datasets or checkpoint binaries are added.
