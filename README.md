# REDIMNET-SV Reproducibility Package

## Margin-Guided Adaptive Model Routing for Resource-Efficient Speaker Verification on Edge Devices

This repository contains the protocol definitions, frozen routing parameters,
per-trial score outputs, and evaluation scripts required to inspect and
reproduce the principal evaluation results reported in the manuscript
**“Margin-Guided Adaptive Model Routing for Resource-Efficient Speaker
Verification on Edge Devices.”**

The package is intentionally evaluation-focused. It does not redistribute
speech recordings, embedding caches, or pretrained model checkpoint binaries.

---

## 1. Scope

The study evaluates a margin-guided conditional-computation strategy using
pretrained ReDimNet models for speaker verification.

The compact ReDimNet-b0 verifier processes every trial. A higher-capacity
ReDimNet-b6 expert is invoked only for trials falling inside a frozen
low-margin routing region.

The principal Frozen-B10 routing policy was developed using VoxCeleb
development data and frozen before final evaluation.

The final VoxCeleb1-O evaluation, USC-SV external-domain evaluation, and
Raspberry Pi 5 deployment evaluation were not used to tune the routing policy.

This package provides evidence for four aspects of the experimental workflow:

1. construction and independence of the DEV/CAL protocols;
2. freezing of decision thresholds and margin-routing policies;
3. per-trial evidence for the principal VoxCeleb1-O and USC-SV evaluations;
4. scripts implementing the protocol and frozen-policy evaluation workflow.

---

## 2. Directory Structure

```text
REDIMNET_SV_REPRODUCIBILITY/
|
|-- README.md
|-- SHA256SUMS.txt
|
|-- protocols/
|   |-- dev_cal_speaker_manifest.csv
|   |-- dev_trials_50k.csv
|   |-- cal_trials_20k.csv
|   |-- r20_protocol_independence_audit.json
|   |-- stage09c_pi5_trial_manifest.csv
|   |-- usc_sv_v1_cal_1E_trials.csv
|   |-- usc_sv_v1_test_1E_trials.csv
|   `-- usc_sv_v1_trials_summary.json
|
|-- policies/
|   |-- cal_frozen_thresholds.json
|   |-- dev_margin_frozen_policies.csv
|   `-- dev_margin_frozen_policies.json
|
|-- scores/
|   |-- dev_b0_b6_scores_50k.csv
|   |-- cal_b0_b6_scores_20k.csv
|   |-- vox1o_frozen_margin_router_trial_audit.csv
|   |-- usc_cal_frozen_b10_scores.csv
|   `-- usc_test_final_frozen_scores.csv
|
`-- evaluation/
    |-- 11_stage03b_create_dev_cal_split.py
    |-- 12_stage03c_create_dev_cal_trials.py
    |-- 17_stage03g1_freeze_cal_thresholds.py
    |-- 21_stage03h3_freeze_margin_policies.py
    `-- 22_stage04a_evaluate_frozen_router_vox1o.py
```

---

## 3. DEV/CAL Protocol

The routing-development protocol was constructed from the VoxCeleb1
development speakers.

The complete development speaker pool contains 1,211 speakers.

A deterministic speaker-disjoint split with seed 42 produced:

- DEV: 969 speakers;
- CAL: 242 speakers;
- speaker overlap between DEV and CAL: 0.

The corresponding verification protocols contain:

- DEV: 50,000 balanced verification trials;
- CAL: 20,000 balanced verification trials.

Target trials use the same speaker with different source sessions, while
impostor trials use different speakers. Identical-path trials and duplicate
unordered trial pairs were excluded.

The protocol definitions are provided in:

```text
protocols/dev_cal_speaker_manifest.csv
protocols/dev_trials_50k.csv
protocols/cal_trials_20k.csv
```

The protocol-independence audit is provided in:

```text
protocols/r20_protocol_independence_audit.json
```

The VoxCeleb1 development speakers used for DEV/CAL are disjoint from the
40 speakers of the VoxCeleb1-O evaluation protocol.

---

## 4. Frozen Routing Policy

The routing signal is based on the decision margin produced by the compact
ReDimNet-b0 verifier.

For a compact-model score `s` and decision threshold `tau`, the margin is

```text
m = |s - tau|.
```

Trials with sufficiently small compact-model margins are considered candidates
for expert evaluation.

The primary Frozen-B10 policy uses the development-derived parameters

```text
tau_b0 = 0.336179674
Delta_B10 = 0.134631842
```

and invokes the ReDimNet-b6 expert when the compact-model score lies inside
the corresponding frozen routing region.

The complete frozen policy definitions are provided in:

```text
policies/dev_margin_frozen_policies.csv
policies/dev_margin_frozen_policies.json
```

Decision thresholds derived from CAL are stored separately in:

```text
policies/cal_frozen_thresholds.json
```

The routing threshold and the verification decision threshold therefore serve
different purposes and should not be interpreted as the same parameter.

Frozen-B10 denotes a development-defined operating region rather than a
requirement that exactly 10% of trials must be routed in every evaluation
domain.

---

## 5. VoxCeleb1-O Evaluation

The principal evaluation uses the standard VoxCeleb1-O protocol containing
37,720 verification trials.

The frozen policy is applied without test-set routing retuning.

The principal reported results are:

| System | Expert usage | EER (%) | minDCF |
|---|---:|---:|---:|
| Static-b0 | 0% | 1.5403 | 0.163998 |
| Frozen-B10 | 9.653% | 0.7238 | 0.135207 |
| Static-b6 | 100% | 0.7158 | 0.054348 |

Thus, Frozen-B10 recovers approximately 99% of the EER improvement from
Static-b0 to Static-b6 while invoking the expert on fewer than 10% of
VoxCeleb1-O trials.

The complete per-trial audit used for this evaluation is provided in:

```text
scores/vox1o_frozen_margin_router_trial_audit.csv
```

This file permits inspection of trial labels, compact/expert scores, routing
decisions, and final routed scores without rerunning neural-network inference.

---

## 6. DEV and CAL Scores

Paired ReDimNet-b0/ReDimNet-b6 scores used during policy development and
calibration are provided in:

```text
scores/dev_b0_b6_scores_50k.csv
scores/cal_b0_b6_scores_20k.csv
```

These files allow the development and calibration stages to be inspected
separately from final evaluation.

No VoxCeleb1-O test scores were used to select the primary routing policy.

---

## 7. USC-SV v1.0 External-Domain Protocol

External-domain evaluation uses the independently defined USC-SV v1.0
speaker-verification protocol derived from the Uzbek Speech Corpus (USC).

The USC-SV protocol was defined independently before its reuse in the present
study.

The primary evaluation uses the 1E protocol.

The supplied protocol files are:

```text
protocols/usc_sv_v1_cal_1E_trials.csv
protocols/usc_sv_v1_test_1E_trials.csv
protocols/usc_sv_v1_trials_summary.json
```

The CAL protocol contains 4,400 trials:

- 400 target trials;
- 4,000 impostor trials.

The TEST protocol contains 5,390 trials:

- 490 target trials;
- 4,900 impostor trials.

CAL and TEST are speaker-disjoint.

The source-domain Frozen-B10 routing parameters are transferred unchanged to
USC-SV. Target-domain CAL is used only for the separate verification decision
thresholds required for fixed-threshold evaluation.

The relevant score files are:

```text
scores/usc_cal_frozen_b10_scores.csv
scores/usc_test_final_frozen_scores.csv
```

On USC-SV TEST, the threshold-swept results are:

| System | EER (%) | minDCF |
|---|---:|---:|
| Static-b0 | 6.6939 | 0.777143 |
| Frozen-B10 | 4.7449 | 0.817755 |
| Static-b6 | 4.6837 | 0.823061 |

Frozen-B10 therefore retains approximately 96.95% of the b0-to-b6 EER
improvement.

The realized expert usage increases to 31.336%, demonstrating that the
source-domain routing budget itself is not portable across domains.

---

## 8. Raspberry Pi 5 Deployment Reproducibility

The repository includes the experimental artifacts used to support the
Raspberry Pi 5 deployment results reported in the manuscript.

The deterministic 500-trial deployment protocol is defined in:

```text
protocols/stage09c_pi5_trial_manifest.csv
```

The `deployment/` directory contains the final reviewer-audited deployment
evidence:

```text
deployment/
|-- latency/
|-- memory/
|-- energy/
|-- lazy_enrollment/
`-- scripts/
```

### 8.1 Cached-Enrollment Latency

The principal deployment benchmark evaluates Static-b0, Static-b6,
Sequential-All, and Frozen-B10 under a cached-enrollment inference scenario.

The Frozen-B10 evaluation contains 500 verification trials:

```text
routed trials     = 40 / 500
expert usage      = 8%
non-routed trials = 460 / 500
```

The corresponding per-trial measurements are provided in:

```text
deployment/latency/r7a_static_b0_trials.csv
deployment/latency/r7a_static_b6_trials.csv
deployment/latency/r7a_sequential_all_trials.csv
deployment/latency/r7a_frozen_b10_trials.csv
deployment/latency/r7a_cached_enrollment_report.json
```

Frozen-B10 achieved a mean end-to-end latency of approximately 1.716 s per
trial, compared with approximately 9.450 s for Static-b6 and 10.413 s for
Sequential-All in this deployment protocol.

The resulting mean-latency reductions were approximately 81.84% relative to
Static-b6 and 83.52% relative to Sequential-All.

The routing decision itself required approximately 0.004 ms per trial.

The supplied per-trial data also distinguish routed and non-routed trials.
The 460 non-routed trials had a mean latency of approximately 0.977 s, whereas
the 40 expert-routed trials had a mean latency of approximately 10.212 s.
Consequently, the upper latency tail is concentrated primarily in the
expert-routed subset.

### 8.2 Routing-Utilization Sensitivity

Deployment sensitivity to increased expert utilization is documented in:

```text
deployment/latency/r7b_routing_usage_sensitivity.csv
deployment/latency/r7b_routing_usage_sensitivity_report.json
```

The experiment evaluates higher realized expert-usage levels and demonstrates
that the latency advantage decreases progressively as more trials invoke the
expert.

These measurements are included to make explicit that the deployment benefit
depends on the realized routing rate rather than being a fixed property of the
model pair.

### 8.3 Static-b2 Edge Baseline

A reviewer-requested ReDimNet-b2 control was evaluated using the same
Raspberry Pi 5 deployment setting.

The corresponding artifacts are:

```text
deployment/latency/r14_static_b2_report.json
deployment/latency/r14_static_b2_trials.csv
deployment/memory/r7c_static_b2_memory.json
deployment/energy/r14c_b2_energy_summary.csv
```

On the 500-trial Raspberry Pi 5 subset, Static-b2 achieved 0.8% EER with a
mean latency of approximately 1.004 s per trial.

Static-b2 therefore represents a strong fixed-capacity edge baseline.
The adaptive Frozen-B10 system should not be interpreted as universally
superior to b2 on every edge-efficiency dimension.

### 8.4 Process-Isolated Memory

Process-isolated peak-memory measurements are provided in:

```text
deployment/memory/r7c_static_b0_memory.json
deployment/memory/r7c_static_b2_memory.json
deployment/memory/r7c_static_b6_memory.json
deployment/memory/r7c_sequential_all_memory.json
deployment/memory/r7c_frozen_b10_memory.json
deployment/memory/r7c_process_isolated_peak_memory_report.json
```

These measurements show that conditional expert invocation reduces average
runtime computation but does not imply a corresponding reduction in peak
process memory when both model capacities remain available to the process.

The manuscript therefore does not claim RAM savings from Frozen-B10.

### 8.5 System-Level Energy

System-level energy consumption was measured at the Raspberry Pi 5 power
input using an AT085 inline USB Type-C power meter.

The final reviewer-audited energy evidence is provided in:

```text
deployment/energy/r21_energy_reproducibility.json
deployment/energy/r21_energy_run_validity.csv
deployment/energy/r21_energy_valid_summary.csv
deployment/energy/r14c_b2_energy_summary.csv
```

Three valid runs per reported configuration were retained in the final
energy analysis.

The principal mean energy measurements were approximately:

| System | Energy per trial (J) |
|---|---:|
| Static-b0 | 5.616 |
| Static-b2 | ~5.76 |
| Static-b6 | 62.688 |
| Sequential-All | 68.328 |
| Frozen-B10 | 10.608 |

Frozen-B10 reduced system-level energy per trial by approximately 83.08%
relative to Static-b6 and 84.47% relative to Sequential-All.

The average power differences between configurations were comparatively
modest; the large energy reductions arise primarily from reduced execution
time.

The precision of the recorded cumulative meter readings should not be
interpreted as instrument uncertainty. In particular, a zero empirical
standard deviation caused by repeated recorded values is not treated as zero
measurement uncertainty.

### 8.6 Lazy Expert-Enrollment Cache

The principal cached-enrollment benchmark precomputes enrollment
representations outside the timed verification loop. To evaluate the
provisioning implications of this assumption, a separate lazy expert-cache
experiment is provided in:

```text
deployment/lazy_enrollment/r27a_lazy_expert_cache_report.json
deployment/lazy_enrollment/r27a_lazy_expert_cache_summary.csv
deployment/lazy_enrollment/r27a_lazy_frozen_b10_trials.csv
```

In this experiment, compact-model enrollment representations are provisioned
initially, whereas expert enrollment representations are created only when an
enrollment identity first reaches the expert.

Only 40 of the 475 unique enrollment utterances required expert enrollment
representations in the evaluated 500-trial sequence.

The resulting provisioning time was approximately 806.53 s, compared with
approximately 4701.08 s for eager provisioning of both model capacities,
corresponding to an approximately 82.84% reduction in initial provisioning
work.

All 40 routed trials were first-use expert-cache misses in this deterministic
sequence. The reported lazy-cache latency therefore represents a conservative
cold-start condition rather than a steady-state cache-hit benchmark.

### 8.7 Deployment Scripts

The public deployment scripts are provided in:

```text
deployment/scripts/
```

They include the final scripts used for:

- cached-enrollment latency evaluation;
- process-isolated memory measurement;
- AT085 energy measurement;
- lazy expert-enrollment evaluation;
- Static-b2 energy control;
- energy-run reproducibility auditing.

Machine-specific absolute paths were removed from the public copies and
replaced with portable relative paths. The original experimental scripts and
the numerical result artifacts were not modified.

### 8.8 Deployment Measurement Boundary

Enrollment-cache construction and warm-up are outside the principal
cached-enrollment latency and energy measurement boundary.

Warm-up uses five files per model and is completed before the measured
experiment begins.

The principal deployment measurements therefore represent a cached-enrollment
verification scenario rather than initial model provisioning.

The lazy expert-enrollment experiment is reported separately to expose the
cost of expert enrollment creation when that assumption is relaxed.

No claim is made that CPU throttling was absent during the experiments because
a preserved throttling-status trace is not available. Recorded temperature
measurements and empirical P95/P99 latency statistics are reported in the
manuscript as deployment diagnostics rather than as evidence of the absence of
thermal effects.

---

## 9. Evaluation Scripts

The `evaluation/` directory preserves the principal scripts corresponding to
the frozen evaluation workflow.

### DEV/CAL speaker split

```text
evaluation/11_stage03b_create_dev_cal_split.py
```

Constructs the deterministic speaker-disjoint DEV/CAL partition.

### DEV/CAL verification trials

```text
evaluation/12_stage03c_create_dev_cal_trials.py
```

Constructs the fixed DEV and CAL verification protocols.

### CAL decision thresholds

```text
evaluation/17_stage03g1_freeze_cal_thresholds.py
```

Derives the verification decision thresholds from CAL.

### Margin-policy freezing

```text
evaluation/21_stage03h3_freeze_margin_policies.py
```

Freezes the development-derived margin-routing policies.

### Final VoxCeleb1-O evaluation

```text
evaluation/22_stage04a_evaluate_frozen_router_vox1o.py
```

Applies the frozen routing policy to VoxCeleb1-O and computes the final
verification metrics.

---

## 10. Pretrained Models

The study uses official pretrained ReDimNet checkpoints.

The primary experiments use the VoxCeleb2 pretraining-only (`vox2/ptn`)
checkpoint family for ReDimNet-b0, ReDimNet-b2, and ReDimNet-b6.

Checkpoint binaries are not redistributed in this package.

The manuscript reports checkpoint identifiers and SHA256 hashes so that the
exact model files used in the experiments can be verified independently.

A separate robustness analysis was also performed with the official
VoxCeleb2 FT-LM checkpoint family. That analysis is reported in the manuscript
as a checkpoint-family robustness test and does not redefine the primary
Frozen-B10 policy.

---

## 11. Dataset Availability

This repository does not redistribute VoxCeleb or USC speech recordings.

VoxCeleb1 and VoxCeleb2 should be obtained from their original distribution
sources subject to their corresponding access and usage conditions.

The Uzbek Speech Corpus (USC) should likewise be obtained from its original
public distribution source under its applicable license.

The protocol files in this package reference the evaluation data but do not
contain the original speech recordings.

---

## 12. Integrity Verification

`SHA256SUMS.txt` contains SHA256 checksums for the scientific artifacts
distributed with this package.

On PowerShell, package integrity can be inspected with:

```powershell
Get-Content SHA256SUMS.txt
```

Individual files can be checked with:

```powershell
Get-FileHash <FILE> -Algorithm SHA256
```

The checksums are intended to make protocol, score, policy, and evaluation-code
versions explicit and auditable.

---

## 13. Reproducibility Levels

Two forms of reproducibility are supported.

### Score-level reproduction

The supplied per-trial score files allow the reported verification metrics,
routing utilization, and routing decisions to be independently recomputed
without neural-network inference.

This is the recommended route for auditing the numerical results reported in
the manuscript.

### Inference-level reproduction

Full inference-level reproduction additionally requires:

- the original speech corpora;
- the official pretrained ReDimNet checkpoints;
- the ReDimNet model implementation;
- the software dependencies used by the evaluation scripts.

These large or externally distributed resources are not duplicated in this
package.

---

## 14. Experimental Separation

The experimental workflow follows the following separation:

```text
VoxCeleb2
    |
    +-- pretrained ReDimNet checkpoints

VoxCeleb1 development speakers
    |
    +-- DEV -> routing-policy development
    |
    `-- CAL -> verification decision calibration

VoxCeleb1-O
    |
    `-- final source-domain evaluation

USC-SV v1.0
    |
    +-- CAL  -> target-domain verification decision calibration
    |
    `-- TEST -> external-domain evaluation

Raspberry Pi 5 subset
    |
    `-- deployment evaluation
```

The final VoxCeleb1-O, USC-SV TEST, and Raspberry Pi 5 evaluation results are
not used to retune the primary source-domain Frozen-B10 routing policy.

---

## 15. Citation

If this reproducibility package is used in academic work, please cite the
associated manuscript:

> *Margin-Guided Adaptive Model Routing for Resource-Efficient Speaker
> Verification on Edge Devices.*

Full bibliographic information will be added after publication.

---

## 16. Contact

Questions concerning the experimental protocol or reproducibility artifacts
should be directed to the corresponding author of the associated manuscript.