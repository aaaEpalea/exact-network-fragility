"""Replace the URL placeholders in the Markdown / CFF files.
usage: python tools/set_repo_url.py https://github.com/<user>/<repo> [https://<your-site>/learnings/<slug>]"""
import sys, glob, os
repo = sys.argv[1].rstrip("/"); site = (sys.argv[2].rstrip("/") if len(sys.argv) > 2 else None); root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
n = 0
for f in glob.glob(root + "/**/*", recursive=True):
    if not f.endswith((".md", ".cff")) or "/node_modules/" in f: continue
    s = open(f, encoding="utf-8").read(); t = s.replace("GITHUB_REPO_URL", repo)
    if site: t = t.replace("WEBSITE_REPORT_URL", site)
    if t != s: open(f, "w", encoding="utf-8").write(t); n += 1
print("updated", n, "files")
