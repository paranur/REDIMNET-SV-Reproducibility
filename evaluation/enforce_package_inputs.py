"""Run one released analysis with external scientific-file reads and networking denied."""
import json,os,runpy,sys
from pathlib import Path
# Import numerical libraries before the file audit so dependency initialization
# does not masquerade as an experimental-data read.
import numpy,sklearn
from sklearn.metrics import roc_curve,roc_auc_score
ROOT=Path(__file__).resolve().parents[1]
script=(ROOT/sys.argv[1]).resolve()
if not script.is_relative_to(ROOT):raise RuntimeError('Script outside package')
allowed=(ROOT.resolve(),Path(sys.base_prefix).resolve(),Path(sys.prefix).resolve())
reads=set()
def audit(event,args):
    if event.startswith('socket.'):raise RuntimeError('Network access denied in score reproduction')
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        name=os.fsdecode(args[0]);p=Path(name).resolve()
        if not any(p.is_relative_to(root) for root in allowed):raise RuntimeError('External file access denied: '+p.name)
        if p.is_relative_to(ROOT):reads.add(p.relative_to(ROOT).as_posix())
sys.addaudithook(audit)
sys.argv=[str(script)]
runpy.run_path(str(script),run_name='__main__')
out=ROOT/'reproduced/access_audit';out.mkdir(parents=True,exist_ok=True)
(out/(script.stem+'.json')).write_text(json.dumps(dict(status='PASS',script=script.relative_to(ROOT).as_posix(),package_files_opened=sorted(reads),external_scientific_reads=0,network_allowed=False),indent=2),encoding='utf-8')
