"""Recompute existing released score-level results; no fitting or inference."""
import csv,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_curve,roc_auc_score
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reproduced/evaluation';OUT.mkdir(parents=True,exist_ok=True)
def read(path):
    p=(ROOT/path).resolve();assert p.is_relative_to(ROOT)
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def js(path):return json.loads((ROOT/path).read_text(encoding='utf-8-sig'))
def metric(y,s):
    fpr,tpr,_=roc_curve(y,s,pos_label=1);fnr=1-tpr;i=int(np.argmin(abs(fnr-fpr)))
    return dict(EER_percent=float(100*(fpr[i]+fnr[i])/2),minDCF=float(np.min(.01*fnr+.99*fpr)/.01),AUC=float(roc_auc_score(y,s)))
def check(actual,expected,tolerance=1e-9):
    for key,value in actual.items():assert abs(value-float(expected[key]))<=tolerance,(key,value,expected[key])
def columns(rows,col):return np.array([float(r[col]) for r in rows])
def align(scores,protocol,enroll='enroll_path',test='test_path'):
    assert len(scores)==len(protocol)
    for s,p in zip(scores,protocol):
        assert int(s['label'])==int(p['label'])
        assert s[enroll].replace('\\','/')==p['enroll_path'].replace('\\','/')
        assert s[test].replace('\\','/')==p['test_path'].replace('\\','/')
        assert int(s['label'])==int(s[enroll].split('/')[0]==s[test].split('/')[0])
def protocol(name):
    return [dict(label=p[0],enroll_path=p[1],test_path=p[2]) for p in (x.split() for x in (ROOT/name).read_text(encoding='utf-8-sig').splitlines() if x.strip())]
def emit(name,rows):
    with (OUT/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
results=[]
def record(dataset,system,y,s,expected,routed=None):
    m=metric(y,s);check(m,expected)
    results.append(dict(dataset=dataset,system=system,trials=len(y),**m,routed_trials='' if routed is None else int(routed)))
base=read('scores/vox1o_b0_b6_scores.csv');p=protocol('protocols/vox1o_veri_test.txt');align(base,p)
audit=read('scores/vox1o_frozen_margin_router_trial_audit.csv');reference=js('evidence/reference/vox1o_frozen_margin_router_results.json')
y=columns(base,'label').astype(int);s0=columns(base,'score_b0');s6=columns(base,'score_b6');route=columns(audit,'route_MARGIN_B10').astype(bool)
for i,(a,b) in enumerate(zip(base,audit)):
    assert int(a['trial_id'])==int(b['trial_id'])==i and a['label']==b['label']
    assert float(a['score_b0'])==float(b['score_b0']) and float(a['score_b6'])==float(b['score_b6'])
record('VoxCeleb1-O','Static-b0',y,s0,reference['static_b0']);record('VoxCeleb1-O','Static-b6',y,s6,reference['static_b6'])
record('VoxCeleb1-O','Frozen-B10',y,np.where(route,s6,s0),next(r for r in reference['policies'] if r['policy_name']=='MARGIN_B10'),route.sum())
# Verify complete DEV/CAL identifiers, speaker disjointness, and both PTN/FT-LM alignments.
speaker_sets=[]
for split,n in [('dev',50000),('cal',20000)]:
    trials=read(f'protocols/{split}_trials_{"50k" if split=="dev" else "20k"}.csv');assert len(trials)==n
    speakers={r[k].split('/')[0] for r in trials for k in ('enroll_path','test_path')};speaker_sets.append(speakers)
    for score_path in [f'scores/{split}_b0_b6_scores_{"50k" if split=="dev" else "20k"}.csv',f'scores/ftlm/{split}_ftlm_b0_b6_scores_{"50k" if split=="dev" else "20k"}.csv']:
        data=read(score_path);align(data,trials)
        assert all(int(a['trial_id'])==int(b['trial_id']) for a,b in zip(data,trials))
assert len(speaker_sets[0])==969 and len(speaker_sets[1])==242 and not speaker_sets[0]&speaker_sets[1]
assert not (speaker_sets[0]|speaker_sets[1])&{r['enroll_path'].split('/')[0] for r in base}
# FT-LM static systems and all six existing frozen policies.
ft=read('scores/ftlm/vox1o_ftlm_b0_b2_b6_scores.csv');align(ft,p,'enroll','test');assert all(int(a['trial_id'])==int(b['trial_id']) for a,b in zip(ft,base))
fy=columns(ft,'label').astype(int);f0=columns(ft,'score_b0');f6=columns(ft,'score_b6');fa=read('scores/ftlm/vox1o_ftlm_frozen_policy_trial_audit.csv')
assert len(fa)==len(ft)==37720
for i,(a,b) in enumerate(zip(ft,fa)):
    assert int(a['trial_id'])==int(b['trial_id'])==i and a['label']==b['label']
    assert float(a['score_b0'])==float(b['score_b0']) and float(a['score_b6'])==float(b['score_b6'])
static=js('evidence/reference/vox1o_ftlm_static_metrics.json')['results']
for model in ('b0','b2','b6'):
    r=next(x for x in static if x['model']=='ReDimNet-'+model)
    record('FT-LM','Static-'+model,fy,columns(ft,'score_'+model),dict(EER_percent=r['EER_percent'],minDCF=r['minDCF'],AUC=r['ROC_AUC']))
expected_ft=read('evidence/reference/vox1o_ftlm_frozen_policy_results.csv')
for pol in read('policies/ftlm_frozen_routing_policies.csv'):
    name=pol['policy'];r=np.abs(f0-float(pol['tau_b0']))<=float(pol['delta_DEV']);mixed=np.where(r,f6,f0)
    assert np.array_equal(r,columns(fa,name+'_route').astype(bool))
    assert np.max(abs(mixed-columns(fa,name+'_mixed_score')))<1e-15
    expected=next(x for x in expected_ft if x['policy']==name);assert int(r.sum())==int(expected['routed_trials'])
    record('FT-LM',name,fy,mixed,expected,r.sum())
# Cleaned protocol uses its own existing score table; do not assume it is a subset.
clean=read('scores/cleaned/cleaned_vox1o_b0_b6_scores.csv');cp=protocol('protocols/vox1o_cleaned_veri_test2.txt');align(clean,cp);assert len(clean)==37611
cy=columns(clean,'label').astype(int);assert cy.sum()==18802
c0=columns(clean,'score_b0');c6=columns(clean,'score_b6');ca=read('scores/cleaned/r24_cleaned_vox1o_trial_audit.csv')
assert len(ca)==len(clean)
for i,(a,b) in enumerate(zip(clean,ca)):
    assert int(a['trial_id'])==int(b['trial_id'])==i and a['label']==b['label']
    assert a['enroll_path']==b['enroll_path'] and a['test_path']==b['test_path']
    assert float(a['score_b0'])==float(b['score_b0']) and float(a['score_b6'])==float(b['score_b6'])
expected=read('evidence/reference/r24_cleaned_vox1o_results.csv');tau=reference['frozen_CAL_tau_b0'];delta=next(r['frozen_delta'] for r in reference['policies'] if r['policy_name']=='MARGIN_B10');cr=np.abs(c0-tau)<=delta
assert np.array_equal(cr,columns(ca,'route_B10').astype(bool))
assert np.array_equal(np.where(cr,c6,c0),columns(ca,'final_score_B10'))
for name,s in [('Static-b0',c0),('Static-b6',c6),('Frozen-B10',np.where(cr,c6,c0))]:
    row=next(r for r in expected if r['system']==name)
    record('cleaned VoxCeleb1-O',name,cy,s,row,int(cr.sum()) if name=='Frozen-B10' else None)
    if name=='Frozen-B10':assert int(cr.sum())==int(row['expert_routed_trials'])
# USC package metadata uses relative roots; values, IDs and trial pairings are unchanged.
for split,n in [('cal',4400),('test',5390)]:
    data=read('scores/usc_cal_frozen_b10_scores.csv' if split=='cal' else 'scores/usc_test_final_frozen_scores.csv')
    trials=read(f'protocols/usc_sv_v1_{split}_1E_trials.csv');assert len(data)==len(trials)==n
    for a,b in zip(data,trials):
        for k in ('label','claimed_speaker','query_speaker','query_index'):assert a[k]==b[k]
        assert a['enroll_path'].replace('\\','/')==b['enroll_1'].replace('\\','/') and a['query_path'].replace('\\','/')==b['query_path'].replace('\\','/')
        assert int(a['label'])==int(a['claimed_speaker']==a['query_speaker'])
    uy=columns(data,'label').astype(int);u0=columns(data,'score_b0');u6=columns(data,'score_b6');ur=columns(data,'route_b6_frozen_b10').astype(bool);u10=columns(data,'score_frozen_b10')
    assert np.array_equal(u10,np.where(ur,u6,u0))
    ref=js('evidence/reference/usc_'+split+'_report.json');metrics=ref['cal_threshold_swept' if split=='cal' else 'threshold_swept_test']
    for name,s in [('b0',u0),('b6',u6),('b10',u10)]:
        r=metrics['frozen_b10' if name=='b10' else name];record('USC '+split,name,uy,s,dict(EER_percent=r['eer_percent'],minDCF=r['minDCF'],AUC=r['roc_auc']),ur.sum() if name=='b10' else None)
    if split=='test':
        for key,operating in ref['cal_frozen_test_operating_points'].items():
            s={'b0':u0,'b6':u6,'b10':u10}[key.split('_')[0]];pred=s>=operating['threshold']
            assert int(((uy==0)&pred).sum())==operating['FA'] and int(((uy==1)&~pred).sum())==operating['MISS']
emit('released_score_metrics.csv',results)
(OUT/'score_validation.json').write_text(json.dumps(dict(status='PASS',metric_rows=len(results),checks=['primary canonical metrics','DEV/CAL trial identities and speaker disjointness','FT-LM full-score alignment, three static systems and six frozen policies','cleaned protocol alignment and three systems','USC CAL/TEST alignment, mixed scores and frozen operating-point counts'],metrics=results),indent=2),encoding='utf-8')
print(json.dumps(dict(status='PASS',metric_rows=len(results),metrics=results),indent=2))
