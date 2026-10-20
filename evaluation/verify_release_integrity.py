"""Verify all released file hashes; reject paths outside this checkout."""
import argparse,csv,hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    manifest=ROOT/'RELEASE_MANIFEST.csv'
    with manifest.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    for r in rows:
        p=(ROOT/r['path']).resolve();assert p.is_relative_to(ROOT) and p.is_file(),r['path']
        assert digest(p)==r['sha256'] and p.stat().st_size==int(r['bytes']),r['path']
    # Manifest includes everything except itself, regenerated outputs and Git state.
    actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and not any(x in p.relative_to(ROOT).parts for x in ('.git','reproduced','__pycache__')) and p.name!='RELEASE_MANIFEST.csv'}
    assert actual=={r['path'] for r in rows},'Manifest completeness mismatch'
    print(json.dumps(dict(status='PASS',verified_files=len(rows),manifest='RELEASE_MANIFEST.csv')))
if __name__=='__main__':main()
