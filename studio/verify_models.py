"""Verify the exact shipped weights without installing UI dependencies."""
import hashlib,json,sys
from pathlib import Path
root=Path(__file__).resolve().parent/'models'
manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
failed=[]
for entry in manifest['files']:
 p=root/entry['file'];print(entry['file'],flush=True)
 if not p.is_file() or p.stat().st_size!=entry['bytes']:
  failed.append(entry['file']);print('  MISSING / SIZE MISMATCH');continue
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 ok=h.hexdigest()==entry['sha256'];print('  OK' if ok else '  HASH MISMATCH')
 if not ok:failed.append(entry['file'])
print('PASS: all 10 models match.' if not failed else 'FAIL: '+', '.join(failed))
sys.exit(bool(failed))
