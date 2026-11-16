import csv, os
rows = []; tasks = ["length", "hist", "reverse", "parity", "majority8", "dyck8"]
def pid(t, c, cat, fb, k): return f"{t}__{c}__{cat}__{fb}__a{k}"
for k in (1, 2, 3):
    for t in tasks:
        for c in ("constrained", "unconstrained"): rows.append((1 if k == 1 else 2, pid(t, c, "3b", "none", k), t, c, "3b", "none", k))
for t in tasks:
    for c in ("constrained", "unconstrained"): rows.append((3, pid(t, c, "3b", "clean_check", 1), t, c, "3b", "clean_check", 1))
for k in (1, 2, 3):
    for t in ("length", "parity"):
        for c in ("constrained", "unconstrained"): rows.append((4, pid(t, c, "4", "none", k), t, c, "4", "none", k))
rows.sort(key=lambda r: (r[0], r[1])); w = csv.writer(open(os.path.join(os.path.dirname(__file__), "run_sheet.csv"), "w")); w.writerow(["priority", "attempt_id", "task", "condition", "category", "feedback", "attempt", "prompt_file"])
for r in rows: w.writerow(list(r) + [f"prompts/rendered/{r[2]}__{r[3]}__{r[4]}__{r[5]}.md"])
print(len(rows), "chats; by priority:", {p: sum(1 for r in rows if r[0] == p) for p in (1, 2, 3, 4)})
