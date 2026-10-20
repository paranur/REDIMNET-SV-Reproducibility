# MR-02 bootstrap design — specified before execution

Primary method: paired speaker-node resampling with single-cluster target weights.

Let S=40 be the number of verified speakers. For each of B=2,000 replicates, sample S speaker IDs uniformly with replacement, using NumPy PCG64 seed 20261020. Let (M_1,...,M_S) be their multinomial multiplicities. For a trial with endpoints (s,t), use

- target (s=t): w_i = M_s;
- impostor (s != t): w_i = M_s M_t.

Use the identical w vector for Static-b0, Static-b6 and Frozen-B10. All target observations for one speaker are one cluster copied M_s times, rather than a diagonal edge copied M_s squared times. Every observed impostor edge is copied once for each combination of copies of its two distinct endpoint speakers. The same multiplicity is used for a speaker in either role and in either class, retaining cross-class dependence. Edges between two copies of the same original speaker are not invented: they are not observed impostor trials. No scores or verification trials are generated. Zero-weight existing trials are excluded only from that replicate. Class totals are normalized separately within each weighted ROC; no extra class-balancing factor is needed.

This is an application of node resampling to a mixed collection of one-speaker clusters and two-speaker observations. The off-diagonal construction is motivated by the multinomial vertex-weight bootstrap described in Bryan Graham, Network Data (2020), section 4.5, pp. 152–154:
httpexternal/NetworkDataPublished.pdf
That source discusses product node weights for distinct-agent dyads; it is not a proof of coverage for this mixed verification protocol or its nonsmooth metrics. The single-cluster target treatment is the design choice required by the observation's one distinct speaker. We choose shared multinomial counts because they give an explicit resampling construction and directly isolate the earlier diagonal-weighting defect. Independent enrollment/test speaker draws would lose dependence for a person appearing in both roles. Unit-weight presence selection discards multiplicities and is retained only as sensitivity evidence.

Metrics follow Stage04A: weighted ROC with tied scores grouped, zero-weight trials removed, and sklearn's default drop_intermediate=True retained. Choose the first ROC point minimizing abs(FNR-FPR); EER=100*(FNR+FPR)/2, with no interpolation. Normalized minDCF=min[0.01*FNR+0.99*FPR]/0.01 (unit costs). Sweeping ROC points is the frozen diagnostic definition of EER/minDCF, not a new deployed decision threshold or policy selection. B10 scores use the original stored routing bits to select b6 or b0 scores. Frozen CAL/DEV values are read only and checked.

Report observed metrics, bootstrap means, standard deviations and 2.5th/97.5th percentiles (NumPy linear quantile). Paired differences are computed within each replicate: B10-b0, B10-b6 and b6-b0. EER differences are percentage points; minDCF differences are absolute normalized units. All 2,000 draws are retained in an audit; invalid replicates are logged, not replaced. Final PASS requires 2,000 valid primary replicates.

Robustness: compare historical R5 (squared target multiplicities) and R13 (unit-weight trial presence). Additionally rerun both using exactly the new primary draws. Run the squared method with canonical ROC pruning, plus the original R5 all-threshold ROC implementation, to separate target weighting, random-seed, and possible metric-implementation effects. These are diagnostics, not candidate selection. Historical R5/R13/R26 files remain untouched and their hashes are checked after execution.

Limitations: only 40 speakers; unbalanced and incomplete observed dyads; conditional on existing recordings, trial protocol, scores and training; no within-speaker recording resampling; assumes dependence is captured by shared speakers and independent disjoint-speaker contributions. Product-weight node bootstrap may be unreliable for weak/degenerate speaker effects. Discrete EER and optimized minDCF are nonsmooth; percentile coverage is approximate, with no finite-sample coverage guarantee for this mixed design. Intervals containing zero do not establish equivalence or noninferiority. No multiple-comparison adjustment or formal noninferiority margin is introduced.
