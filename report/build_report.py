import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))   # repository root
import re, os, sys, glob
sys.path.insert(0, ".")
import overlay_AF as A, overlay_GM as G
ROOT = REPO; R = open(f"{ROOT}/RESULTS.md").read().split("\n")
PARTS = {**A.PARTS, **G.PARTS}; SUB = {**A.SUB, **G.SUB}; FIG = {}
for d in (A.FIG, G.FIG): FIG.update({k: [v] for k, v in d.items()})
addfig_defined = True
def addfig(k, f, cap): FIG.setdefault(k, []).append((f, cap))
addfig("I3", "I2_nscale.png", "**Figure I3c.** Minimum key width divided by the number of stored transitions against the number of states n (random automata, one and four programs): there is no compression gain as n grows.")
addfig("D8", "diagram_tied.png", "**Figure D8.** Weight-tied design: the transition matrices (up, down, hold) are shared; each program only selects which role each token uses.")
addfig("I3", "diagram_fwi.png", "**Figure I3b.** The fast-weight interpreter: a key built from token, state and program address (Hadamard product of random codes) reads the fast transition memory F; the clip restores the state; readout and initial state are also fast memories.")
addfig("I5", "I0_certificate_gap.png", "**Figure I5b.** Certified radius against first failure found by the attack, for the tied designs, the interpreter and the LLM-built transformers. Every point lies on or above the diagonal; the vertical distance to it is the looseness of the certificate.")
addfig("SYN", "X1_overview_construction.png", "**Figure S1.** Absolute noise tolerance (sigma50) for every network in the consolidated dataset, by construction method. Compiled and LLM-built transformers lie in the same broad range; restoring-state designs are an order of magnitude higher; the within-method spread is large.")
addfig("L2" ,"L2_first_order_vs_attack.png", "**Figure L2.** First-order L-infinity radius divided by the attack-found radius for the 12 development networks, with autodiff gradients (relu'(0)=0) and with the one-sided derivative (relu'(0)=1). A ratio of 1 means the first-order model predicts the attack exactly.")
addfig("L5", "L1_predictors.png", "**Figure L5.** Predicted against measured sigma50 for predictor v3 (left) and v4 (right) on three construction methods. v4 brings the LLM-built networks and the bilinear networks close to the diagonal; the compiled `length` and `hist` programs stay far above it (the predictor is too optimistic).")
addfig("K2", "K1_sigma50_by_task.png", "**Figure K1.** Absolute sigma50 for every attempt, by task and condition: fresh run (dots), contaminated run (diamonds) and the Tracr compiled baseline (squares, shared tasks only).")
def demote(text, by):
    out = []
    for l in text.split("\n"):
        m = re.match(r"^(#+) (.*)", l)
        out.append(("#" * (len(m.group(1)) + by) + " " + m.group(2)) if m else l)
    return "\n".join(out)

def cells(l): return [c.strip() for c in l.strip().strip("|").split("|")]
def mk(rows): return ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])] + ["| " + " | ".join(r) + " |" for r in rows[1:]]
def normalize(tab):
    head = cells(tab[0]); K = len(head); rows = [head]
    for l in tab[2:]:
        c = cells(l)
        if len(c) > K: c = [" · ".join(c[: len(c) - K + 1])] + c[len(c) - K + 1:]
        elif len(c) < K: c = c + [""] * (K - len(c))
        rows.append(c)
    return ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * K] + ["| " + " | ".join(r) + " |" for r in rows[1:]]
def transform_table(tab):
    tab = normalize(tab); head = cells(tab[0]); body = [cells(l) for l in tab[2:]]
    if head == ["path", "value"]:
        d = {}
        for p, v in body:
            m = re.match(r"^(.*?) / (.*)$", p); k, met = (m.group(1), m.group(2)) if m else (p, "value"); d.setdefault(k, {})[met] = v
        cols = []
        [cols.append(c) for r in d.values() for c in r if c not in cols]
        if len(cols) <= 14: return mk([["case"] + cols] + [[k] + [r.get(c, "") for c in cols] for k, r in d.items()])
        return tab
    if head[0] == "item" and "s50(curve)" in head and "params" in head:
        meta = {}; met = {}; order = []; cols = []
        for r in body:
            if " / " in r[0]:
                prog, m_ = r[0].split(" / ", 1); v = [x for x in r[1:] if x][-1] if any(r[1:]) else ""; met.setdefault(prog, {})[m_] = v
                if m_ not in cols: cols.append(m_)
            else: meta[r[0]] = r[1:]; order.append(r[0])
        mc = [h for h in head[1:] if h != "s50(curve)"]
        return mk([["program"] + mc + cols] + [[p] + [meta[p][head.index(h) - 1] for h in mc] + [met.get(p, {}).get(c, "") for c in cols] for p in order])
    if head[0] == "item" and body and all(" / " in r[0] for r in body):
        return mk([["program", "variant"] + head[1:]] + [r[0].split(" / ", 1) + r[1:] for r in body])
    return tab

def render_body(lines):
    out, i = [], 0
    while i < len(lines):
        l = lines[i]
        if l.startswith("|"):
            j = i
            while j < len(lines) and lines[j].startswith("|"): j += 1
            tab = transform_table(lines[i:j]) if j - i > 6 else normalize(lines[i:j])
            if len(tab) > 32: out += ["", f"<details><summary>Full table ({len(tab)-2} rows)</summary>", ""] + tab + ["", "</details>", ""]
            else: out += [""] + tab + [""]
            i = j; continue
        out.append(l); i += 1
    return "\n".join(out)
# ---- parse RESULTS.md
secs = {}; order = []; cur = None; part = None; partlines = {}
for l in R:
    m = re.match(r"^## ([A-M])\. (.*)", l)
    if m: part = m.group(1); partlines[part] = {"title": m.group(2), "intro": [], "subs": []}; cur = None; continue
    m = re.match(r"^### ([A-M]\d+)\. (.*)", l)
    if m and part: cur = {"id": m.group(1), "title": m.group(2), "lines": []}; partlines[part]["subs"].append(cur); continue
    if re.match(r"^## ", l): part = None; cur = None; continue
    if part:
        (cur["lines"] if cur else partlines[part]["intro"]).append(l)
def figs(k):
    s = ""
    for f, cap in FIG.get(k, []):
        if os.path.exists(f"figures/{f}"): s += f"\n![{cap.split('**')[1] if '**' in cap else f}](figures/{f})\n\n{cap}\n"
        else: s += f"\n<!-- missing figure {f} -->\n"
    return s
missing = []
def sub_block(sub, level="###"):
    sid = sub["id"]; q, t = SUB.get(sid, (None, None)); s = f"\n{level} {sid}. {sub['title']}\n\n"
    if q is None: missing.append(sid)
    else: s += f"**Question.** {q}\n\n"
    s += render_body([x for x in sub["lines"]]).strip("\n") + "\n"
    if t: s += f"\n> **Takeaway.** {t}\n"
    s += figs(sid); return s
parts_out = {}
for p in "ABCDEFGHIJK":
    d = partlines[p]; title, intro = PARTS[p]; s = f"\n## Part {p}. {title}\n\n{intro}\n"
    extra = "\n".join(x for x in d["intro"] if x.strip())
    if extra: s += "\n" + render_body(d["intro"]).strip("\n") + "\n"
    for sub in d["subs"]: s += sub_block(sub)
    parts_out[p] = s
rd = lambda p: open(f"{ROOT}/{p}").read()
# ---- Part L from theory documents
L = f"\n## Part L. {PARTS['L'][0]}\n\n{PARTS['L'][1]}\n\n"
L += "### L1. Pre-registration and the first predictors (gradient-based, v1 and v2)\n\n" + demote(rd("theory/PREREG_linear_predictor.md"), 3) + "\n\n" + demote(rd("theory/THEORY_EXPLORATION_1.md"), 3) + "\n"
L += "\n### L2. Pre-registered cliff experiment and the kink artifact\n\n" + demote(rd("theory/PREREG_cliff_experiment.md"), 3) + "\n" + figs("L2")
L += "\n### L3. Predictor v3 (tangent-cone) and its held-out test\n\n" + demote(rd("theory/THEORY_EXPLORATION_2.md"), 3) + "\n"
L += "\n### L4. Cross-construction test of v3 (Tracr and bilinear networks)\n\n" + demote(rd("theory/PREREG_cross_construction.md"), 3) + "\n\n" + demote(rd("theory/THEORY_EXPLORATION_3.md"), 3) + "\n"
L += "\n### L5. Predictor v4 (tangent-cone plus saturation margins)\n\n" + demote(rd("theory/PREREG_v4_saturation.md"), 3) + "\n\n" + demote(rd("theory/THEORY_EXPLORATION_4.md"), 3) + "\n" + figs("L5")
L += "\n### L6. Theory extraction: consolidated status\n\n" + demote(rd("THEORY_EXTRACTION.md"), 3) + "\n"
L += "\n### L7. Candidate relationships considered\n\n" + demote(rd("THEORY_CANDIDATES.md").replace("||k||_1","norm1(k)").replace("||k||_2","norm2(k)").replace("(|w| ~","(abs(w) ~"), 3) + "\n"
parts_out["L"] = L
M = f"\n## Part M. {PARTS['M'][0]}\n\n{PARTS['M'][1]}\n\n" + demote(rd("PRIOR_ART_AUDIT.md"), 2) + "\n\n### M-notes. Earlier quick notes on the LLM-synthesis question\n\n" + demote(rd("PRIOR_ART_NOTES.md"), 3) + "\n"
parts_out["M"] = M
tail = open("src/90_tail.md").read(); syn_fig = figs("SYN")
tail = tail.replace("### What the evidence supports", "### What the evidence supports\n\n" + syn_fig.strip() + "\n", 1) if syn_fig else tail
appC = "\n## Appendix C. Protocol and pre-registration documents\n\n### C1. LLM-synthesis protocol (frozen before any attempt)\n\n" + demote(rd("llm_synthesis/PROTOCOL.md"), 2) + "\n\n### C2. Fresh-session instructions\n\n" + demote(rd("llm_synthesis/README_FRESH_SESSION.md"), 2) + "\n\n### C3. Harness changelog\n\n" + demote(rd("llm_synthesis/HARNESS_CHANGELOG.md"), 2) + "\n"
appD = "\n## Appendix D. Figures and data index\n\n" + "\n".join(f"* `figures/{os.path.basename(f)}`" for f in sorted(glob.glob("figures/*.png"))) + "\n\nThe consolidated dataset is `results/implementation_dataset.csv`; per-experiment JSON and logs are in `results/`.\n"
os.makedirs("parts", exist_ok=True)
for p, s in parts_out.items(): open(f"parts/part_{p}.md", "w").write(s)
full = open("src/00_front.md").read() + "\n\n" + open("src/01_background.md").read() + "".join(parts_out[p] for p in "ABCDEFGHIJKLM") + "\n\n" + tail + appC + appD + "\n\n" + open("src/99_refs.md").read()
open("REPORT.md", "w").write(full)
print("chars", len(full), "words", len(full.split()), "| subsections without overlay:", missing)
print({p: len(parts_out[p].split()) for p in parts_out})
print("figure refs:", len(re.findall(r"!\[", full)), "missing figure markers:", len(re.findall("missing figure", full)), "details blocks:", full.count("<details>"))
