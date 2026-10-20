"""Execute all portable analyses, compare expected outputs and audit package isolation."""
import csv,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reproduced';OUT.mkdir(exist_ok=True)
commands=[]
for script in ('evaluate_released_scores.py','recalculate_independent_expert_benefit.py','corrected_speaker_dependent_paired_bootstrap.py'):
    cmd=[sys.executable,'evaluation/enforce_package_inputs.py','evaluation/'+script]
    result=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    # Runtime logs remain ignored; they are not release artifacts.
    (OUT/(script+'.log')).write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
    assert result.returncode==0,script+' failed: '+result.stderr
    commands.append('python evaluation/enforce_package_inputs.py evaluation/'+script)
    print('PASS '+script,flush=True)
comparisons=[]
for task in ('mr01','mr02'):
    for expected in sorted((ROOT/'evidence'/task).glob('*.csv')):
        actual=ROOT/'reproduced'/task/expected.name
        assert actual.is_file(),expected.name
        with expected.open(encoding='utf-8-sig',newline='') as f:a=list(csv.DictReader(f))
        with actual.open(encoding='utf-8-sig',newline='') as f:b=list(csv.DictReader(f))
        assert len(a)==len(b),(task,expected.name,'row count')
        for left,right in zip(a,b):
            assert left.keys()==right.keys()
            for key in left:
                if left[key]==right[key]:continue
                try:assert abs(float(left[key])-float(right[key]))<=1e-12
                except ValueError:raise AssertionError((task,expected.name,key))
        comparisons.append(dict(task=task,file=expected.name,rows=len(a),status='PASS'))
for script in ('evaluate_released_scores','recalculate_independent_expert_benefit','corrected_speaker_dependent_paired_bootstrap'):
    report=json.loads((OUT/'access_audit'/(script+'.json')).read_text())
    assert report['external_scientific_reads']==0
report=dict(status='PASS',commands=commands,expected_output_comparisons=comparisons,score_evaluation=json.loads((OUT/'evaluation/score_validation.json').read_text()),package_only_scientific_reads=True)
(OUT/'package_reproduction_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(dict(status='PASS',expected_tables=len(comparisons),metric_rows=report['score_evaluation']['metric_rows'],external_scientific_reads=0),indent=2))
