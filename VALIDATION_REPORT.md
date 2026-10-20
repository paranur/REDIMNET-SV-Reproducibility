# Executed release validation



Status: PASS (2026-10-20, Asia/Tashkent). Baseline public commit: adc1f366453bf4ef737c374c0a3a614e9300d59e.



Executed with existing CPython 3.10.0, NumPy 2.2.6 and scikit-learn 1.7.2; no installation or environment creation. The local package-only suite used these exact repository-relative commands:



```sh

python evaluation/run_package_reproduction.py

python evaluation/check_public_release_privacy.py

python evaluation/verify_release_integrity.py

```



The reproduction runner executes the three portable analysis scripts through evaluation/enforce_package_inputs.py. Its file audit rejects scientific inputs outside the package, allows the installed interpreter/dependency tree, and rejects network operations. Zero external scientific-file reads were observed. All 14 expected CSV output tables matched within absolute numerical tolerance 1e-12, including per-trial labels, 2,000 bootstrap draws, paired intervals, sensitivity replicates and speaker weights. This is score-level re-execution, not new scientific experimentation.



MR-01: all 37,720 canonical trials aligned on IDs, labels, scores and both recording endpoints. Potential benefit=396; potential harm=100; compact errors=683; benefit ROC-AUC=0.9799983816294288; AP=0.29831855727109813. Actual B10 corrections=378, harms=102, net=276, routed=3641. Potential labels are independent of routing.



MR-02: all 2,000 primary replicates valid; seed 20261020; target weights M_s; distinct-speaker impostor weights M_s*M_t; pairing across b0/b6/B10. All saved marginal/paired confidence intervals and replicate tables reproduce. The original cached ROC implementation passes 270 checks against sklearn and literal integer replication. EER intervals including zero do not establish equivalence or noninferiority.



Existing score results: 21 metric rows reproduced across original PTN, FT-LM static systems and all six frozen policies, cleaned VoxCeleb1-O, and USC CAL/TEST. Primary and cleaned trial endpoints and labels were checked against the original released official lists. FT-LM test scores align exactly with the same 37,720-trial protocol. DEV/CAL trial endpoints and IDs align with PTN and FT-LM scores, and speaker counts 969/242 are disjoint from one another and the test speakers. Frozen routing bits and mixed scores reproduce their archived audits. USC labels, speaker IDs, query indices and relative recording suffixes align across protocols/scores; TEST frozen-threshold FA/MISS counts reproduce.



FT-LM EER-B10: 3773 routed; EER=0.5567338282078461%; minDCF=0.07529162248144215. Cleaned Frozen-B10: 37611 trials, 3604 routed; EER=0.5716417898039743%; minDCF=0.1327400774181432.



Original scientific inputs were rehashed after preparation: all unchanged. Existing public metadata normalizations preserve every non-path CSV field; original/public hashes are retained. The privacy scanner rejects audio/checkpoint/embedding/archive/private-key file types, unexpected binary content, machine roots and common credential patterns. It scans only the current release tree; original Git history is retained. It is a targeted scan, not an exhaustive secret-detection or legal guarantee. License/provenance review is limited to the author-owned score-level release and identifier metadata described in LICENSE_NOTES.md; no new blanket license is asserted.



NOT REPRODUCIBLE from this package alone: original audio inference, training, checkpoint export/conversion, embedding construction, hardware remeasurement and private upstream USC source-manifest reconstruction. No such capability is claimed. Original speech, embeddings, checkpoint binaries, private datasets, credentials, private machine roots, the manuscript and reviewer responses are excluded.



The authoritative exact file inventory and SHA-256 values are RELEASE_MANIFEST.csv; original source hashes are in provenance/source_artifact_manifest.csv. A clean-checkout test is recorded in the local experiment journal before publication. Re-run the commands above to independently verify the released tree.

