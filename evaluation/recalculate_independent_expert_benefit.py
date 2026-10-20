"""MR-01: score-only analysis; Python standard library; no inference or fitting."""
import csv
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reproduced/mr01"
OUT.mkdir(parents=True, exist_ok=True)
COMMAND = 'python evaluation/recalculate_independent_expert_benefit.py'
SOURCES = {
 'scores': ('results/stage01_clean_vox1o/scores/vox1o_b0_b6_scores.csv', 'Stage01, scripts/02_compute_b0_b6_scores.py'),
 'audit': ('results/stage04_final_test/stage04a_frozen_router/vox1o_frozen_margin_router_trial_audit.csv', 'Stage04A, scripts/22_stage04a_evaluate_frozen_router_vox1o.py'),
 'thresholds': ('results/stage03_dev_cal/stage03g_thresholds/cal_frozen_thresholds.json', 'Stage03G-1, CAL-only freeze'),
 'threshold_csv': ('results/stage03_dev_cal/stage03g_thresholds/cal_frozen_thresholds.csv', 'Stage03G-1; Stage04A input'),
 'policy': ('results/stage03_dev_cal/stage03h_policy_freeze/dev_margin_frozen_policies.json', 'Stage03H-3, DEV-only freeze'),
 'mixed_threshold': ('results/reviewer_audit/r2b_fixed_decision_threshold_transfer/r2b_cal_frozen_decision_thresholds.json', 'R2B, CAL-only mixed decision freeze'),
 'cal_scores': ('results/stage03_dev_cal/stage03f_scores/cal_b0_b6_scores_20k.csv', 'Stage03F CAL; hash pinned in R2B'),
 'history': ('results/reviewer_audit/r15_frozen_decision_expert_benefit/r15_report.json', 'R15 historical report'),
 'history_trials': ('results/reviewer_audit/r15_frozen_decision_expert_benefit/r15_vox1o_per_trial.csv', 'R15 historical per-trial decisions'),
 'protocol': ('voxceleb1/veri_test.txt', 'Canonical VoxCeleb1-O protocol'),
}
for name in ['02_compute_b0_b6_scores.py','17_stage03g1_freeze_cal_thresholds.py','21_stage03h3_freeze_margin_policies.py','22_stage04a_evaluate_frozen_router_vox1o.py','36_stage_r2b_fixed_decision_threshold_transfer.py','67_stage_r15_frozen_decision_expert_benefit.py']:
    SOURCES[name] = ('scripts/' + name, 'Provenance code, read only; not executed')

SOURCES = {'scores': ('scores/vox1o_b0_b6_scores.csv', 'Stage01, scripts/02_compute_b0_b6_scores.py'), 'audit': ('scores/vox1o_frozen_margin_router_trial_audit.csv', 'Stage04A, scripts/22_stage04a_evaluate_frozen_router_vox1o.py'), 'thresholds': ('policies/cal_frozen_thresholds.json', 'Stage03G-1, CAL-only freeze'), 'threshold_csv': ('policies/cal_frozen_thresholds.csv', 'Stage03G-1; Stage04A input'), 'policy': ('policies/dev_margin_frozen_policies.json', 'Stage03H-3, DEV-only freeze'), 'mixed_threshold': ('policies/r2b_cal_frozen_decision_thresholds.json', 'R2B, CAL-only mixed decision freeze'), 'cal_scores': ('scores/cal_b0_b6_scores_20k.csv', 'Stage03F CAL; hash pinned in R2B'), 'history': ('evidence/historical/r15_frozen_decision_expert_benefit/r15_report.json', 'R15 historical report'), 'history_trials': ('evidence/historical/r15_frozen_decision_expert_benefit/r15_vox1o_per_trial.csv', 'R15 historical per-trial decisions'), 'protocol': ('protocols/vox1o_veri_test.txt', 'Canonical VoxCeleb1-O protocol'), '02_compute_b0_b6_scores.py': ('provenance/code/02_compute_b0_b6_scores.py', 'Provenance code, read only; not executed'), '17_stage03g1_freeze_cal_thresholds.py': ('provenance/code/17_stage03g1_freeze_cal_thresholds.py', 'Provenance code, read only; not executed'), '21_stage03h3_freeze_margin_policies.py': ('provenance/code/21_stage03h3_freeze_margin_policies.py', 'Provenance code, read only; not executed'), '22_stage04a_evaluate_frozen_router_vox1o.py': ('provenance/code/22_stage04a_evaluate_frozen_router_vox1o.py', 'Provenance code, read only; not executed'), '36_stage_r2b_fixed_decision_threshold_transfer.py': ('provenance/code/36_stage_r2b_fixed_decision_threshold_transfer.py', 'Provenance code, read only; not executed'), '67_stage_r15_frozen_decision_expert_benefit.py': ('provenance/code/67_stage_r15_frozen_decision_expert_benefit.py', 'Provenance code, read only; not executed')}

def path(key):
    p=(ROOT/SOURCES[key][0]).resolve()
    if not p.is_relative_to(ROOT.resolve()): raise RuntimeError("Input outside package")
    return p
def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
def read(key):
    with path(key).open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))
def js(key): return json.loads(path(key).read_text(encoding='utf-8-sig'))
def write_csv(name, rows):
    with (OUT / name).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def require(condition, message):
    if not condition: raise RuntimeError(message)

def ranking(labels, margins):
    # Equal margins form one threshold group. ROC gives half credit for ties;
    # AP uses grouped step integration; trapezoidal PR is additionally reported.
    pos = sum(labels); neg = len(labels)-pos
    tp = fp = 0; auc = ap = trap = 0.; old_r = 0.; old_p = 1.
    for _, group in itertools.groupby(sorted(zip(margins,labels)), key=lambda x:x[0]):
        g = list(group); p = sum(y for _,y in g); q = len(g)-p
        auc += q*(tp+p/2)/(pos*neg)
        tp += p; fp += q
        r = tp/pos; precision = tp/(tp+fp)
        ap += (r-old_r)*precision
        trap += (r-old_r)*(old_p+precision)/2
        old_r,old_p = r,precision
    return auc,ap,trap

def main():
    manifest = [{'role':k,'path':str(path(k).resolve()),'stage':v[1],'sha256':sha(path(k))} for k,v in SOURCES.items()]
    write_csv('source_manifest.csv', manifest)
    hashes = {r['role']:r['sha256'] for r in manifest}
    require(hashes['scores']=='f71d954a7e237f8d43140f03df132d88cabb9e6cbdf5659fd92c94055d38ee4c','Original TEST score hash differs from R2B pinned hash')
    require(hashes['cal_scores']=='d48855e7c95a58a52c79d315afac5332fb30d9484fdbffdfa73783343c246cdf','Original CAL score hash differs from R2B pinned hash')
    scores,audit,history = read('scores'),read('audit'),read('history_trials')
    protocol = [line.split() for line in path('protocol').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    require(len(scores)==len(audit)==len(history)==len(protocol)==37720,'Trial count mismatch')
    thresholds = js('thresholds'); policy = js('policy'); mixed = js('mixed_threshold'); historical = js('history')
    require(thresholds['source_split']=='CAL' and thresholds['source_trials']==20000,'CAL provenance')
    t0,t6 = [thresholds['models'][m]['frozen_primary_EER']['threshold'] for m in ('b0','b6')]
    tm = mixed['final_decision_thresholds_from_CAL']['EER_oriented']
    delta = next(p['delta_margin_cutoff'] for p in policy['policies'] if p['policy_name']=='MARGIN_B10')
    require(policy['status']=='FROZEN' and policy['dev_trials']==50000,'DEV provenance')
    require(abs(tm-historical['frozen_parameters']['tau_b10_mixed'])<1e-12,'Mixed threshold provenance mismatch')
    threshold_rows=read('threshold_csv')
    for m,t in [('b0',t0),('b6',t6)]:
        row=next(r for r in threshold_rows if r['model'].lower() in (m,'redimnet-'+m))
        require(abs(float(row['frozen_EER_threshold'])-t)<1e-15,'Threshold CSV/JSON mismatch')
    # Gate all alignment and provenance checks before calculating benefit labels.
    max_margin_error=0.
    for i,(s,a,h,p) in enumerate(zip(scores,audit,history,protocol)):
        require(int(s['trial_id'])==int(a['trial_id'])==int(h['trial_id'])==i,'Trial ID/order mismatch')
        require(s['label']==a['label']==h['label']==p[0],'Label mismatch')
        require([s['enroll_path'],s['test_path']]==p[1:]==[h['enroll_path'],h['test_path']],'Endpoint mismatch')
        require(int(s['label'])==int(s['enroll_path'].split('/')[0]==s['test_path'].split('/')[0]),'Speaker/label mismatch')
        for c in ('score_b0','score_b6'):
            require(math.isfinite(float(s[c])) and float(s[c])==float(a[c])==float(h[c]),'Score mismatch')
        margin=abs(float(s['score_b0'])-t0)
        err=abs(margin-float(a['b0_abs_margin_frozen_tau'])); max_margin_error=max(max_margin_error,err)
        require(err<1e-14,'Frozen margin mismatch')
        require(int(a['route_MARGIN_B10'])==int(h['route_b10'])==int(margin<=delta),'Frozen route mismatch')
    rows=[]
    for s,a,h in zip(scores,audit,history):
        y=int(s['label']); b0=float(s['score_b0'])>=t0; b6=float(s['score_b6'])>=t6
        c0=b0==y; c6=b6==y
        # Independent labels use every trial, with no routing input.
        benefit=(not c0) and c6; harm=c0 and (not c6)
        route=int(a['route_MARGIN_B10'])
        pred=(float(s['score_b6']) if route else float(s['score_b0']))>=tm
        actual_b=(not c0) and pred==y; actual_h=c0 and pred!=y
        require(int(b0)==int(h['b0_pred']) and int(b6)==int(h['b6_pred']) and int(pred)==int(h['mixed_pred']),'Historical decision mismatch')
        require(int(actual_b)==int(h['benefit']) and int(actual_h)==int(h['harm']),'Historical transition mismatch')
        rows.append(dict(trial_id=int(s['trial_id']),label=y,b0_pred=int(b0),b6_pred=int(b6),potential_benefit=int(benefit),potential_harm=int(harm),b0_error=int(not c0),margin=float(a['b0_abs_margin_frozen_tau']),routed=route,mixed_pred=int(pred),actual_correction=int(actual_b),actual_harm=int(actual_h)))
    n=len(rows); nb=sum(r['potential_benefit'] for r in rows); nh=sum(r['potential_harm'] for r in rows)
    roc,ap,trap=ranking([r['potential_benefit'] for r in rows],[r['margin'] for r in rows])
    metrics=dict(n_trials=n,potential_benefit=nb,potential_harm=nh,compact_errors=sum(r['b0_error'] for r in rows),benefit_prevalence=nb/n,benefit_roc_auc=roc,benefit_pr_auc_average_precision=ap,benefit_pr_auc_trapezoidal=trap,ap_lift_over_prevalence=ap/(nb/n))
    ranked=sorted(rows,key=lambda r:(r['margin'],r['trial_id']))
    regions=[]
    for pct in (5,10,20,30):
        k=n*pct//100; region=ranked[:k]; b=sum(r['potential_benefit'] for r in region); h=sum(r['potential_harm'] for r in region)
        regions.append(dict(lowest_margin_percent=pct,n_trials=k,benefit_count=b,benefit_recall=b/nb,harm_count=h,harm_capture=h/nh,benefit_precision=b/k,lift=(b/k)/(nb/n),boundary_margin=region[-1]['margin'],boundary_ties_total=sum(r['margin']==region[-1]['margin'] for r in rows),boundary_ties_included=sum(r['margin']==region[-1]['margin'] for r in region)))
    executed=[]
    for group,subset in [('all',rows),('target',[r for r in rows if r['label']==1]),('impostor',[r for r in rows if r['label']==0])]:
        b=sum(r['actual_correction'] for r in subset); h=sum(r['actual_harm'] for r in subset); routed=sum(r['routed'] for r in subset)
        executed.append(dict(group=group,n_trials=len(subset),routed_trials=routed,routed_utilization=routed/len(subset),actual_corrections=b,actual_harms=h,net_corrected=b-h,nonrouted_corrections=sum(r['actual_correction'] for r in subset if not r['routed']),nonrouted_harms=sum(r['actual_harm'] for r in subset if not r['routed'])))
    e=executed[0]; hist=historical['point_estimates']
    checks={'routed_n':e['routed_trials'],'benefit_n':e['actual_corrections'],'harm_n':e['actual_harms'],'net_corrected_decisions':e['net_corrected'],'benefit_FA_corrected':executed[2]['actual_corrections'],'benefit_MISS_corrected':executed[1]['actual_corrections'],'harm_FA_created':executed[2]['actual_harms'],'harm_MISS_created':executed[1]['actual_harms']}
    for key,value in checks.items(): require(value==hist[key],f'Historical R15 mismatch: {key}')
    # Independent implementation also reproduces historical grouped ranking values.
    oldroc,oldap,_=ranking([r['actual_correction'] for r in rows],[r['margin'] for r in rows])
    require(abs(oldroc-hist['benefit_roc_auc'])<1e-12 and abs(oldap-hist['benefit_pr_auc'])<1e-12,'Ranking implementation differs from R15 reference')
    for source in manifest: require(sha(Path(source['path']))==source['sha256'],'Input changed during execution')
    write_csv('summary_metrics.csv',[metrics]); write_csv('margin_region_capture.csv',regions)
    write_csv('actual_routing_transitions.csv',executed)
    write_csv('per_trial_independent_labels_and_transitions.csv',[{k:v for k,v in r.items() if k!='margin'} for r in rows])
    params=dict(tau_b0=t0,tau_b6=t6,tau_B10_mixed=tm,delta_B10=delta)
    validations=[('All 37,720 trials included','PASS'),('Potential benefit independent of routing','PASS'),('Frozen thresholds unchanged','PASS'),('Exact score, label, ID and endpoint alignment','PASS'),('No TEST-driven policy selection','PASS'),('Executed-policy counts reproduce R15','PASS'),('Previous circularity resolved for independent ranking','PASS')]
    report='MR-01 independent held-out expert-benefit recalculation\n\nFinal decision: PASS\n\n'
    report+='Execution: `'+COMMAND+'`\n\nPython: '+sys.version+'\n\nFrozen parameters: '+json.dumps(params)+'\n\n'
    report+='Definitions: decision = score >= respective CAL threshold. Potential benefit = b0 wrong AND b6 correct; potential harm = b0 correct AND b6 wrong, on ALL trials before any route-conditioned analysis. Ranking score is negative stored b0 absolute margin.\n\n'
    report+='PR-AUC primary estimator is grouped average precision (step integration), consistent with R15. Trapezoidal PR area is separately reported. ROC uses grouped ties (half credit); region membership sorts by (stored margin, trial_id), selecting floor(N*p/100). These descriptive regions do not define or select deployment policies. Lift = AP / benefit prevalence; region lift is precision / prevalence.\n\n'
    report+='Source integrity: TEST and CAL score SHA-256 reproduce hashes pinned in R2B. Every trial ID, label, b0/b6 score, and protocol endpoint agrees across original scores, Stage04A and R15 where present. Labels additionally agree with speaker identities. Threshold CSV and JSON agree. Original DEV policy reproduces every stored B10 route. Maximum stored-margin reconstruction error: '+str(max_margin_error)+'. All source hashes unchanged after analysis. See source_manifest.csv for exact paths, hashes and stages.\n\n'
    report+='Independent resultexternal/n'+json.dumps(metrics,indent=2)+'\n```\n\nMargin regionexternal/n'+json.dumps(regions,indent=2)+'\n```\n\nExecuted B10:\n```json\n'+json.dumps(executed,indent=2)+'\n```\n\n'
    report+='Target corrections are false-reject corrections; impostor corrections are false-accept corrections. Target harms create false rejects; impostor harms create false accepts. Executed decisions use the CAL-frozen MIXED threshold on routed b6/nonrouted b0 scores, rather than substituting static b6 decisions. Nonrouted transitions are explicitly counted.\n\n'
    report+='Historical R15: all per-trial static/mixed decisions, transitions, route bits and eight aggregate counts reproduce exactly. Historical ROC/AP also reproduce to 1e-12. Independent ranking differs because its labels represent static expert potential on every trial; historical labels represent executed mixed-system corrections. Historical results remain untouched.\n\n'
    report+='Validatioexternal/n'+''.join('- '+name+': '+status+'\n' for name,status in validations)+'\n'
    report+='Limitations: descriptive point estimates on one fixed held-out protocol, with dependent speaker trials; no new confidence intervals or causal/generalization claims. Available frozen artifact and provenance records are verified, without rerunning model inference or re-estimating CAL thresholds. Ranking assesses potential correctness benefit at these operating points, not a newly optimized policy. No retraining, audio inference, package installation, recalibration or manuscript access occurred.\n'
    (OUT/'validation_report.md').write_text(report,encoding='utf-8')
    print(json.dumps(dict(metrics=metrics,regions=regions,executed=executed,status='PASS'),indent=2))

if __name__=='__main__':
    try: main()
    except Exception as exc:
        (OUT/'validation_report.md').write_text('MR-01 FINAL DECISION: FAIL\n\nExecution stopped: '+repr(exc)+'\nNo PASS claimed. Resolve provenance/alignment before proceeding.\n',encoding='utf-8')
        raise
