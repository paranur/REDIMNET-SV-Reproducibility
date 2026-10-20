"""Check the current public tree for excluded binary artifacts and private roots."""
import json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
patterns={
 'absolute machine root':re.compile(r'(?i)(?:\b[A-Z]:[\\/]|/(?:home|Users|mnt|media)/)'),
 'private key':re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
 'GitHub token':re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})'),
 'AWS key':re.compile(r'AKIA[A-Z0-9]{16}'),
 'credential URL':re.compile(r'https?://[^\s/]+:[^\s/]+@'),
}
excluded={'.wav','.flac','.mp3','.mp4','.pt','.pth','.ckpt','.npy','.npz','.pkl','.pickle','.zip','.7z','.tar','.gz','.sqlite','.env','.pem','.key'}
checked=0;failures=[]
for p in ROOT.rglob('*'):
    if not p.is_file() or any(s in p.relative_to(ROOT).parts for s in ('.git','reproduced','__pycache__')):continue
    rel=p.relative_to(ROOT).as_posix();checked+=1
    if p.suffix.lower() in excluded:failures.append(dict(path=rel,rule='excluded binary/secret file type'));continue
    try:text=p.read_bytes().decode('utf-8-sig')
    except UnicodeDecodeError:failures.append(dict(path=rel,rule='unexpected binary content'));continue
    for rule,pattern in patterns.items():
        if pattern.search(text):failures.append(dict(path=rel,rule=rule))
print(json.dumps(dict(status='FAIL' if failures else 'PASS',checked_files=checked,failures=failures,scope='current release tree; Git history and ignored runtime outputs excluded'),indent=2))
if failures:raise SystemExit(1)
