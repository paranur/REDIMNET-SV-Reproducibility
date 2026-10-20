"""MR-02: paired speaker-node bootstrap using existing frozen scores only."""
from __future__ import annotations
import csv
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import sklearn
from sklearn.metrics import roc_curve

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT / "reproduced/mr02"
OUT.mkdir(parents=True, exist_ok=True)
SEED=20261020
REPLICATES=2000
PYTHON=Path(sys.executable)
COMMAND='python evaluation/corrected_speaker_dependent_paired_bootstrap.py'
SOURCES={
 'canonical_scores':('results/stage01_clean_vox1o/scores/vox1o_b0_b6_scores.csv','Stage01 frozen cosine scores'),
 'protocol':('voxceleb1/veri_test.txt','Canonical VoxCeleb1-O protocol'),
 'routing_audit':('results/stage04_final_test/stage04a_frozen_router/vox1o_frozen_margin_router_trial_audit.csv','Stage04A frozen scores/margins/routes'),
 'canonical_metrics':('results/stage04_final_test/stage04a_frozen_router/vox1o_frozen_margin_router_results.json','Stage04A observed metrics'),
 'static_thresholds':('results/stage03_dev_cal/stage03g_thresholds/cal_frozen_thresholds.json','Stage03G-1 CAL freeze'),
 'static_threshold_csv':('results/stage03_dev_cal/stage03g_thresholds/cal_frozen_thresholds.csv','Stage03G-1 input used by Stage04A'),
 'frozen_policy':('results/stage03_dev_cal/stage03h_policy_freeze/dev_margin_frozen_policies.json','Stage03H-3 DEV freeze'),
 'frozen_policy_csv':('results/stage03_dev_cal/stage03h_policy_freeze/dev_margin_frozen_policies.csv','Stage03H-3 input used by Stage04A'),
 'deployed_decision_threshold':('results/reviewer_audit/r2b_fixed_decision_threshold_transfer/r2b_cal_frozen_decision_thresholds.json','R2B CAL-derived mixed threshold; read only, not needed for ROC metrics'),
 'previous_trial_audit':('results/reviewer_audit/r15_frozen_decision_expert_benefit/r15_vox1o_per_trial.csv','R15 archived mixed scores and speaker endpoints'),
 'score_generation_code':('scripts/02_compute_b0_b6_scores.py','Provenance code; not executed'),
 'canonical_metric_code':('scripts/22_stage04a_evaluate_frozen_router_vox1o.py','Canonical EER/minDCF definitions; not executed'),
 'historical_R5_code':('scripts/43_stage_r5a_dependency_aware_bootstrap.py','Historical multiplicity-weighted code; not executed'),
 'historical_R13_code':('scripts/63_stage_r13_bootstrap_sensitivity.py','Historical presence-sensitivity code; not executed'),
 'historical_R26_code':('scripts/80_stage_r26_usc_paired_speaker_bootstrap.py','Historical USC code; not executed'),
}
for directory in ('r5a_dependency_aware_bootstrap','r13_bootstrap_sensitivity','r26_usc_paired_speaker_bootstrap'):
    for p in sorted((ROOT/'evidence/historical'/directory).rglob('*')):
        if p.is_file(): SOURCES[str(p.relative_to(ROOT))]=(str(p.relative_to(ROOT)),f'Historical {directory}; preserved')

SOURCES = {'canonical_scores': ('scores/vox1o_b0_b6_scores.csv', 'Stage01 frozen cosine scores'), 'protocol': ('protocols/vox1o_veri_test.txt', 'Canonical VoxCeleb1-O protocol'), 'routing_audit': ('scores/vox1o_frozen_margin_router_trial_audit.csv', 'Stage04A frozen scores/margins/routes'), 'canonical_metrics': ('evidence/reference/vox1o_frozen_margin_router_results.json', 'Stage04A observed metrics'), 'static_thresholds': ('policies/cal_frozen_thresholds.json', 'Stage03G-1 CAL freeze'), 'static_threshold_csv': ('policies/cal_frozen_thresholds.csv', 'Stage03G-1 input used by Stage04A'), 'frozen_policy': ('policies/dev_margin_frozen_policies.json', 'Stage03H-3 DEV freeze'), 'frozen_policy_csv': ('policies/dev_margin_frozen_policies.csv', 'Stage03H-3 input used by Stage04A'), 'deployed_decision_threshold': ('policies/r2b_cal_frozen_decision_thresholds.json', 'R2B CAL-derived mixed threshold; read only, not needed for ROC metrics'), 'previous_trial_audit': ('evidence/historical/r15_frozen_decision_expert_benefit/r15_vox1o_per_trial.csv', 'R15 archived mixed scores and speaker endpoints'), 'score_generation_code': ('provenance/code/02_compute_b0_b6_scores.py', 'Provenance code; not executed'), 'canonical_metric_code': ('provenance/code/22_stage04a_evaluate_frozen_router_vox1o.py', 'Canonical EER/minDCF definitions; not executed'), 'historical_R5_code': ('provenance/code/43_stage_r5a_dependency_aware_bootstrap.py', 'Historical multiplicity-weighted code; not executed'), 'historical_R13_code': ('provenance/code/63_stage_r13_bootstrap_sensitivity.py', 'Historical presence-sensitivity code; not executed'), 'historical_R26_code': ('provenance/code/80_stage_r26_usc_paired_speaker_bootstrap.py', 'Historical USC code; not executed'), 'results\\reviewer_audit\\r5a_dependency_aware_bootstrap\\r5a_dependency_aware_bootstrap_report.json': ('evidence/historical/r5a_dependency_aware_bootstrap/r5a_dependency_aware_bootstrap_report.json', 'Historical r5a_dependency_aware_bootstrap; preserved'), 'results\\reviewer_audit\\r5a_dependency_aware_bootstrap\\r5a_dependency_aware_ci_summary.csv': ('evidence/historical/r5a_dependency_aware_bootstrap/r5a_dependency_aware_ci_summary.csv', 'Historical r5a_dependency_aware_bootstrap; preserved'), 'results\\reviewer_audit\\r5a_dependency_aware_bootstrap\\r5a_speaker_cluster_bootstrap_replicates.csv': ('evidence/historical/r5a_dependency_aware_bootstrap/r5a_speaker_cluster_bootstrap_replicates.csv', 'Historical r5a_dependency_aware_bootstrap; preserved'), 'results\\reviewer_audit\\r13_bootstrap_sensitivity\\r13_bootstrap_replicates.csv': ('evidence/historical/r13_bootstrap_sensitivity/r13_bootstrap_replicates.csv', 'Historical r13_bootstrap_sensitivity; preserved'), 'results\\reviewer_audit\\r13_bootstrap_sensitivity\\r13_bootstrap_report.json': ('evidence/historical/r13_bootstrap_sensitivity/r13_bootstrap_report.json', 'Historical r13_bootstrap_sensitivity; preserved'), 'results\\reviewer_audit\\r13_bootstrap_sensitivity\\r13_bootstrap_summary.csv': ('evidence/historical/r13_bootstrap_sensitivity/r13_bootstrap_summary.csv', 'Historical r13_bootstrap_sensitivity; preserved'), 'results\\reviewer_audit\\r26_usc_paired_speaker_bootstrap\\r26_bootstrap_samples.csv': ('evidence/historical/r26_usc_paired_speaker_bootstrap/r26_bootstrap_samples.csv', 'Historical r26_usc_paired_speaker_bootstrap; preserved'), 'results\\reviewer_audit\\r26_usc_paired_speaker_bootstrap\\r26_summary.json': ('evidence/historical/r26_usc_paired_speaker_bootstrap/r26_summary.json', 'Historical r26_usc_paired_speaker_bootstrap; preserved')}

def source(key):
    p=(ROOT/SOURCES[key][0]).resolve()
    if not p.is_relative_to(ROOT.resolve()): raise RuntimeError("Input outside package")
    return p
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def read_json(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def write_csv(name,rows):
    with (OUT/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def require(condition,message):
    if not condition: raise RuntimeError(message)
def speaker(p): return p.replace('\\','/').split('/')[0]
def timestamp(): return datetime.now(timezone(timedelta(hours=5))).isoformat(timespec='seconds')

def weights_for(multiplicity,enroll,test,target,method):
    if method=='presence': return ((multiplicity[enroll]>0)|(multiplicity[test]>0)).astype(np.int64)
    weights=multiplicity[enroll]*multiplicity[test]
    if method=='corrected': weights[target]=multiplicity[enroll[target]]
    return weights

class CachedRoc:
    """Pre-sort frozen scores. Rebuild every weighted ROC, including zero removal."""
    def __init__(self,labels,scores):
        self.order=np.argsort(-scores,kind='mergesort')
        self.labels=labels[self.order]
        sorted_scores=scores[self.order]
        self.ends=np.r_[np.flatnonzero(np.diff(sorted_scores)),len(scores)-1]
    def curve(self,weights):
        w=weights[self.order]
        tp=np.cumsum(w*self.labels,dtype=np.int64)[self.ends]
        fp=np.cumsum(w*(1-self.labels),dtype=np.int64)[self.ends]
        # roc_curve excludes zero-weight observations before grouping scores.
        keep=np.diff(np.r_[0,tp+fp])>0
        tp,fp=tp[keep],fp[keep]
        require(len(tp)>0 and tp[-1]>0 and fp[-1]>0,'Missing class in replicate')
        return tp,fp
    def metrics(self,weights,prune=True):
        tp,fp=self.curve(weights)
        if prune and len(fp)>2:
            keep=np.r_[True,np.logical_or(np.diff(fp,2),np.diff(tp,2)),True]
            tp,fp=tp[keep],fp[keep]
        tpr=np.r_[0.,tp/tp[-1]]; fpr=np.r_[0.,fp/fp[-1]]
        fnr=1.-tpr
        index=int(np.argmin(np.abs(fnr-fpr)))
        return float(100.*(fpr[index]+fnr[index])/2.),float(np.min(.01*fnr+.99*fpr)/.01)

def reference_metrics(y,s,w,prune=True):
    fpr,tpr,_=roc_curve(y,s,sample_weight=w,drop_intermediate=prune)
    fnr=1-tpr; idx=int(np.argmin(np.abs(fnr-fpr)))
    return np.array([100*(fpr[idx]+fnr[idx])/2,np.min(.01*fnr+.99*fpr)/.01])
def percentile_row(values):
    x=np.asarray(values,dtype=float); x=x[np.isfinite(x)]
    lo,hi=np.quantile(x,[.025,.975],method='linear')
    return dict(bootstrap_mean=float(x.mean()),bootstrap_sd=float(x.std(ddof=1)),ci95_low=float(lo),ci95_high=float(hi),valid_replicates=int(len(x)))
def markdown_table(rows):
    keys=list(rows[0]); return '| '+' | '.join(keys)+' |\n| '+' | '.join(['---']*len(keys))+' |\n'+''.join('| '+' | '.join(f'{r[k]:.9f}' if isinstance(r[k],float) else str(r[k]) for k in keys)+' |\n' for r in rows)


def main():
    started=timestamp(); tic=time.monotonic()
    require((ROOT/'evidence/mr02/bootstrap_design.md').is_file(),'Design must be documented before bootstrap execution')
    design_hash=digest(ROOT/'evidence/mr02/bootstrap_design.md')
    manifest=[dict(role=k,path=str(source(k).resolve()),origin=v[1],sha256=digest(source(k))) for k,v in SOURCES.items()]
    write_csv('source_manifest.csv',manifest)
    require(next(r['sha256'] for r in manifest if r['role']=='canonical_scores')=='f71d954a7e237f8d43140f03df132d88cabb9e6cbdf5659fd92c94055d38ee4c','Canonical score hash differs from historical pinned hash')
    scores=read_csv(source('canonical_scores')); audit=read_csv(source('routing_audit')); previous=read_csv(source('previous_trial_audit'))
    protocol=[line.split() for line in source('protocol').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    require(len(scores)==len(audit)==len(protocol)==len(previous)==37720,'Trial count mismatch')
    for i,(s,a,p,h) in enumerate(zip(scores,audit,protocol,previous)):
        require(int(s['trial_id'])==int(a['trial_id'])==int(h['trial_id'])==i,'Trial ID/order mismatch')
        require(s['label']==a['label']==p[0]==h['label'],'Label/order mismatch')
        require([s['enroll_path'],s['test_path']]==p[1:]==[h['enroll_path'],h['test_path']],'Trial endpoint mismatch')
        require(int(s['label'])==int(speaker(p[1])==speaker(p[2])),'Speaker identities inconsistent with labels')
        require(speaker(s['enroll_path'])==h['enroll_spk'] and speaker(s['test_path'])==h['test_spk'],'Historical speaker identity mismatch')
        for c in ('score_b0','score_b6'): require(float(s[c])==float(a[c])==float(h[c]),'Score alignment mismatch')
    labels=np.array([int(s['label']) for s in scores],dtype=np.int64); target=labels==1
    require(int(target.sum())==18860 and int((~target).sum())==18860,'Class counts mismatch')
    speakers=sorted({speaker(s[c]) for s in scores for c in ('enroll_path','test_path')})
    require(len(speakers)==40,'Expected 40 verified speakers')
    lookup={s:i for i,s in enumerate(speakers)}
    enroll=np.array([lookup[speaker(s['enroll_path'])] for s in scores]); test=np.array([lookup[speaker(s['test_path'])] for s in scores])
    require(np.array_equal(enroll==test,target),'Target/impostor identity mismatch')
    static=read_json(source('static_thresholds')); policies=read_json(source('frozen_policy'))
    tau=static['models']['b0']['frozen_primary_EER']['threshold']
    delta=next(p['delta_margin_cutoff'] for p in policies['policies'] if p['policy_name']=='MARGIN_B10')
    require(static['source_split']=='CAL' and policies['status']=='FROZEN' and policies['dev_trials']==50000,'Freeze provenance mismatch')
    threshold_csv=read_csv(source('static_threshold_csv')); policy_csv=read_csv(source('frozen_policy_csv'))
    require(abs(float(next(r['frozen_EER_threshold'] for r in threshold_csv if r['model'].lower() in ('b0','redimnet-b0')))-tau)<1e-15,'CAL CSV/JSON mismatch')
    require(abs(float(next(r['delta_margin_cutoff'] for r in policy_csv if r['policy_name']=='MARGIN_B10'))-delta)<1e-15,'DEV CSV/JSON mismatch')
    b0=np.array([float(s['score_b0']) for s in scores]); b6=np.array([float(s['score_b6']) for s in scores])
    require(np.isfinite(b0).all() and np.isfinite(b6).all(),'Nonfinite scores')
    route=np.array([int(a['route_MARGIN_B10']) for a in audit],dtype=bool)
    require(np.array_equal(route,np.abs(b0-tau)<=delta),'Frozen routes differ from freeze records')
    require(np.array_equal(route,[int(h['route_b10'])==1 for h in previous]),'Historical route mismatch')
    require(np.max(np.abs(np.abs(b0-tau)-[float(a['b0_abs_margin_frozen_tau']) for a in audit]))<1e-14,'Frozen margin mismatch')
    b10=np.where(route,b6,b0)
    require(np.array_equal(b10,[float(h['mixed_score']) for h in previous]),'Frozen mixed scores differ from R15')
    require(int(route.sum())==3641,'Frozen route count mismatch')
    names=('b0','b6','B10'); arrays=(b0,b6,b10); caches=[CachedRoc(labels,s) for s in arrays]
    original=read_json(source('canonical_metrics'))
    canonical=[original['static_b0'],original['static_b6'],next(p for p in original['policies'] if p['policy_name']=='MARGIN_B10')]
    observed=np.array([cache.metrics(np.ones(len(labels),dtype=np.int64)) for cache in caches])
    reference_checks=0
    for k,(s,expected) in enumerate(zip(arrays,canonical)):
        require(np.allclose(observed[k],[expected['EER_percent'],expected['minDCF']],rtol=0,atol=1e-12),'Full sample canonical metrics mismatch')
        require(np.allclose(observed[k],reference_metrics(labels,s,np.ones(len(labels),dtype=int)),rtol=0,atol=1e-12),'Cached ROC differs from sklearn')
        reference_checks+=1
    full=[dict(system=name,n_trials=37720,n_target=18860,n_impostor=18860,EER_percent=float(observed[k,0]),normalized_minDCF=float(observed[k,1])) for k,name in enumerate(names)]
    write_csv('full_sample_metrics.csv',full)
    write_csv('verified_trial_speaker_mapping.csv',[dict(trial_id=i,label=int(labels[i]),enroll_speaker=speakers[enroll[i]],test_speaker=speakers[test[i]],route_B10=int(route[i])) for i in range(len(labels))])
    # Explicit weighting tests: arbitrary count vector, repeated-node target, and endpoint symmetry.
    probe=np.arange(40,dtype=np.int64)%5
    weights=weights_for(probe,enroll,test,target,'corrected')
    require(np.array_equal(weights[target],probe[enroll[target]]),'Target multiplicity used more than once')
    require(np.array_equal(weights[~target],probe[enroll[~target]]*probe[test[~target]]),'Both impostor endpoints not represented')
    require(np.array_equal(weights,weights_for(probe,test,enroll,target,'corrected')),'Endpoint symmetry violated')
    doubled=np.zeros(40,dtype=np.int64); doubled[enroll[0]]=2
    require(np.all(weights_for(doubled,enroll,test,target,'corrected')[target & (enroll==enroll[0])]==2),'Target M=2 must have weight 2, not 4')
    # Directly change each endpoint for an observed impostor to verify both contribute.
    imp=int(np.flatnonzero(~target)[0]); probe2=np.ones(40,dtype=np.int64)
    probe2[enroll[imp]]=2; probe2[test[imp]]=3
    require(weights_for(probe2,enroll,test,target,'corrected')[imp]==6,'Distinct impostor M=2,3 must have weight 6')
    rng=np.random.default_rng(SEED)
    draws=np.array([np.bincount(rng.integers(0,40,size=40),minlength=40) for _ in range(REPLICATES)],dtype=np.int64)
    require(np.all(draws.sum(axis=1)==40),'Invalid speaker draws')
    write_csv('speaker_multiplicities.csv',[dict(replicate=i,**{s:int(draws[i,k]) for k,s in enumerate(speakers)}) for i in range(REPLICATES)])
    methods=('corrected','squared_canonical','squared_historical_ROC','presence')
    values={m:np.full((REPLICATES,3,2),np.nan) for m in methods}
    replicate_rows=[]; validity=[]; primary_target_check=True; both_endpoint_check=True
    for r,multiplicity in enumerate(draws):
        corrected=weights_for(multiplicity,enroll,test,target,'corrected')
        squared=weights_for(multiplicity,enroll,test,target,'squared')
        presence=weights_for(multiplicity,enroll,test,target,'presence')
        primary_target_check &= np.array_equal(corrected[target],multiplicity[enroll[target]])
        both_endpoint_check &= np.array_equal(corrected[~target],multiplicity[enroll[~target]]*multiplicity[test[~target]])
        for method,w,prune in [('corrected',corrected,True),('squared_canonical',squared,True),('squared_historical_ROC',squared,False),('presence',presence,True)]:
            ptotal=int(w[target].sum()); ntotal=int(w[~target].sum()); valid=ptotal>0 and ntotal>0
            row=dict(method=method,replicate=r,valid=int(valid),target_weight=ptotal,impostor_weight=ntotal,positive_weight_trials=int((w>0).sum()))
            validity.append(dict(method=method,replicate=r,valid=int(valid),reason='' if valid else 'zero class weight'))
            if valid:
                for k,cache in enumerate(caches):
                    values[method][r,k]=cache.metrics(w,prune=prune)
                    if r<3 or r%100==0:
                        require(np.allclose(values[method][r,k],reference_metrics(labels,arrays[k],w,prune),rtol=0,atol=1e-12),'Weighted cached/reference ROC mismatch')
                        reference_checks+=1
                    # Integer-weight equivalence to literal replication of EXISTING trials.
                    if r==0 and method=='corrected':
                        indices=np.repeat(np.arange(len(w)),w)
                        expected=reference_metrics(labels[indices],arrays[k][indices],np.ones(len(indices),dtype=int))
                        require(np.allclose(values[method][r,k],expected,rtol=0,atol=1e-12),'Weighted/literal replication mismatch')
                        reference_checks+=1
                    row[names[k]+'_EER_percent']=float(values[method][r,k,0]); row[names[k]+'_normalized_minDCF']=float(values[method][r,k,1])
            replicate_rows.append(row)
        if (r+1)%200==0: print(f'Completed {r+1}/{REPLICATES} paired draws; elapsed {time.monotonic()-tic:.1f}s',flush=True)
    require(primary_target_check and both_endpoint_check,'Replicate-level dependency checks failed')
    require(all(np.isfinite(v).all() for v in values.values()),'Some requested replicates invalid; see validation audit')
    write_csv('bootstrap_replicates.csv',replicate_rows); write_csv('replicate_validity.csv',validity)
    summary=[]; paired=[]
    for method in methods:
        v=values[method]
        for k,name in enumerate(names):
            for j,metric in enumerate(('EER_pp','normalized_minDCF')):
                summary.append(dict(method=method,system=name,metric=metric,observed=float(observed[k,j]),**percentile_row(v[:,k,j])))
        for left,right in ((2,0),(2,1),(1,0)):
            for j,metric in enumerate(('EER_pp','normalized_minDCF')):
                paired.append(dict(method=method,comparison=names[left]+' minus '+names[right],metric=metric,observed_difference=float(observed[left,j]-observed[right,j]),**percentile_row(v[:,left,j]-v[:,right,j])))
    write_csv('bootstrap_summary.csv',[r for r in summary if r['method']=='corrected'])
    write_csv('paired_confidence_intervals.csv',[r for r in paired if r['method']=='corrected'])
    write_csv('sensitivity_bootstrap_summary.csv',[r for r in summary if r['method']!='corrected'])
    write_csv('sensitivity_paired_confidence_intervals.csv',[r for r in paired if r['method']!='corrected'])
    historical=[]
    for label,relative in [('R5','r5a_dependency_aware_bootstrap/r5a_dependency_aware_ci_summary.csv'),('R13','r13_bootstrap_sensitivity/r13_bootstrap_summary.csv')]:
        for r in read_csv(ROOT/'evidence/historical'/relative):
            if r['metric'].startswith('delta_'):
                historical.append(dict(method=label,comparison='B10 minus b6',metric='EER_pp' if 'eer' in r['metric'] else 'normalized_minDCF',bootstrap_mean=float(r.get('mean',r.get('bootstrap_mean'))),ci95_low=float(r['ci95_low']),ci95_high=float(r['ci95_high'])))
    comparison=historical+[dict(method=r['method'],comparison=r['comparison'],metric=r['metric'],bootstrap_mean=r['bootstrap_mean'],ci95_low=r['ci95_low'],ci95_high=r['ci95_high']) for r in paired if r['comparison']=='B10 minus b6']
    write_csv('historical_comparison.csv',comparison)
    # Verify all source artifacts, including every R5/R13/R26 file, remain byte-identical.
    for r in manifest: require(digest(Path(r['path']))==r['sha256'],'Original artifact changed during execution: '+r['path'])
    require(digest(ROOT/'evidence/mr02/bootstrap_design.md')==design_hash,'Pre-execution design changed during execution')
    validations=[
      ('Exactly 37,720 canonical trials, 18,860 per class','PASS'),
      ('Both speaker identities verified against protocol, labels and R15; 40 speakers','PASS'),
      ('Target weight uses one speaker multiplicity, checked on all draws','PASS'),
      ('Impostor weight uses both distinct endpoints, checked on all draws','PASS'),
      ('No squared target multiplicity; explicit M=2 gives weight 2 test','PASS'),
      ('Identical trial weights and speaker draws across all three systems','PASS'),
      (f'Canonical metric definitions; {reference_checks} reference metric checks; literal integer replication checks','PASS'),
      ('All 2,000 primary replicates valid; no redraws, skips or replacements','PASS'),
      ('No TEST-driven routing policy or model selection','PASS'),
      ('Original source and historical R5/R13/R26 SHA-256 hashes unchanged','PASS'),
    ]
    primary=[r for r in paired if r['method']=='corrected']
    def interval_interpret(r):
        if r['ci95_high']<0: return 'lower than comparator'
        if r['ci95_low']>0: return 'higher than comparator'
        return 'interval includes zero; equivalence and noninferiority not established'
    conclusions=[dict(comparison=r['comparison'],metric=r['metric'],interpretation=interval_interpret(r)) for r in primary]
    status_eer=next(r for r in primary if r['comparison']=='B10 minus b6' and r['metric']=='EER_pp')
    status_dcf=next(r for r in primary if r['comparison']=='B10 minus b6' and r['metric']=='normalized_minDCF')
    unchanged=(status_eer['ci95_low']<=0<=status_eer['ci95_high'] and status_dcf['ci95_low']>0)
    text='MR-02 corrected speaker-dependent paired bootstrap\n\nFinal decision: PASS\n\n'
    text+=f'Started: {started}; completed: {timestamp()} (Asia/Tashkent).\n\nCommand: `{COMMAND}`\n\n'
    text+=f'Runtime: Python {platform.python_version()}, NumPy {np.__version__}, scikit-learn {sklearn.__version__}. No packages installed. PCG64 seed={SEED}; B={REPLICATES}; S=40; P_target=.01, unit costs.\n\n'
    text+='Reviewer concern: applying endpoint multiplicities indiscriminately squares the weight of a same-speaker target observation. Primary weights are M_s for targets and M_s*M_t for distinct-speaker impostors, shared across all systems. Full pre-execution mathematical rationale, references and limitations: bootstrap_design.md.\n\n'
    text+=f'Frozen routing tau_b0={tau!r}, delta_B10={delta!r}; routed=3641/37720. B10 frozen mixed scores match archived R15 mixed_score exactly. Static b6 CAL threshold and mixed decision threshold are preserved but are not inputs to threshold-swept EER/minDCF. All input hashes unchanged.\n\n'
    text+='Full sample metricexternal/n'+markdown_table(full)+'\nPrimary marginal intervalexternal/n'+markdown_table([r for r in summary if r['method']=='corrected'])+'\nPrimary paired confidence intervalexternal/n'+markdown_table(primary)
    text+='\nHistorical and same-draw robustness comparison (B10 minus b6):\n\n'+markdown_table(comparison)
    text+='\nHistorical R5 and R13 use seed 20260921; new primary and sensitivity methods use identical seed-20261020 draws. R5 is the original squared-multiplicity approach; R13 is unit-weight presence sensitivity, not a full multiplicity bootstrap. The squared_historical_ROC diagnostic reproduces the original all-threshold weighted ROC definition; squared_canonical uses Stage04A pruning. The new same-draw comparison separates the target-weight correction from the random-seed change and ROC pruning. Historical results are not overwritten. R26 uses USC rather than VoxCeleb1-O, so no cross-dataset numerical comparison is made; its artifacts are hash-protected.\n'
    text+='\nScientific conclusions relative to historical R5/R13 B10-vs-b6: '+('UNCHANGED' if unchanged else 'CHANGED')+'.\n\n'+markdown_table(conclusions)
    text+='\nLower EER/minDCF is better. CIs containing zero do not demonstrate equivalence or noninferiority. Marginal CI overlap is not used to infer paired differences. These are approximate pointwise percentile intervals, without simultaneous-comparison correction. Full-sample observed differences are distinguished from bootstrap means.\n\nValidatioexternal/n'+''.join('- '+name+': '+result+'\n' for name,result in validations)
    text+='\nLimitations: 40 speakers, fixed unbalanced protocol and recordings, approximate percentile coverage for discrete/nonsmooth metrics, nondegenerate speaker-effect assumption, no retraining or trial/audio resampling, no finite-sample guarantee. The correction fixes target over-weighting; it does not prove universal bootstrap validity. See pre-execution design for full limitations and primary-source statistical context.\n'
    (OUT/'bootstrap_validation_report.md').write_text(text,encoding='utf-8')
    metadata=dict(status='PASS',started=started,completed=timestamp(),command=COMMAND,seed=SEED,replicates=REPLICATES,speakers=speakers,design_sha256=design_hash,script_sha256=digest(Path(__file__)),python=sys.version,numpy=np.__version__,sklearn=sklearn.__version__,reference_metric_checks=reference_checks,valid_primary_replicates=2000,invalid_primary_replicates=0,conclusions_relative_to_R5_R13='UNCHANGED' if unchanged else 'CHANGED',validations=validations)
    (OUT/'execution_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    artifacts=sorted(p.name for p in OUT.iterdir() if p.is_file())
    print(json.dumps(dict(status='PASS',full_sample=full,paired=primary,historical_comparison=comparison,conclusions_relative_to_R5_R13=metadata['conclusions_relative_to_R5_R13'],elapsed_seconds=time.monotonic()-tic),indent=2),flush=True)

if __name__=='__main__':
    try: main()
    except Exception as exc:
        report='MR-02 FINAL DECISION: FAIL\n\nStopped at '+timestamp()+': '+repr(exc)+'\nNo PASS claimed; resolve the failure before proceeding.\n'
        (OUT/'bootstrap_validation_report.md').write_text(report,encoding='utf-8')
        raise
