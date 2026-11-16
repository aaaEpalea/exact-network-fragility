"""Writes FREEZE.json: sha256 of every file of the package (harness, prompts, protocol). Run BEFORE any attempt; re-run `python freeze.py --check` to verify nothing changed."""
import hashlib, json, os, sys
root = os.path.dirname(os.path.abspath(__file__)); files = []
for d, _, fs in os.walk(root):
    if any(x in d for x in ("submissions", "records", "results", "__pycache__")): continue
    for f in fs:
        if f in ("FREEZE.json",) or f.endswith(".pyc"): continue
        files.append(os.path.relpath(os.path.join(d, f), root))
files.sort(); cur = {f: hashlib.sha256(open(os.path.join(root, f), "rb").read()).hexdigest() for f in files}
if "--check" in sys.argv:
    old = json.load(open(os.path.join(root, "FREEZE.json")))["files"]; bad = [f for f in set(old) | set(cur) if old.get(f) != cur.get(f)]; print("package unchanged" if not bad else f"CHANGED: {bad}")
else:
    pk = hashlib.sha256(json.dumps(cur, sort_keys=True).encode()).hexdigest(); json.dump(dict(package_sha256=pk, files=cur), open(os.path.join(root, "FREEZE.json"), "w"), indent=1); print("package sha256:", pk)
