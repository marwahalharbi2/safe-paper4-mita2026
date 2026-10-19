"""Compare a run of safe_paper4.py + strict_f1.py against the published Fig. 2, and redraw it.

Usage: python scripts/check_and_plot.py results/results.json results/fig2_reproduced.png
Exits non-zero if any reproduced number differs from the paper by more than 0.01.
"""
import json
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, __file__.rsplit("scripts", 1)[0] + "scripts")
from strict_f1 import load, strict  # noqa: E402

# Published values, Paper 4 Fig. 2 (P, R, F1 in %; standalone SAFE in %)
PAPER = {
    "Sentence-T5":              dict(sa=78.63, Soft=(97.39, 73.07, 83.49), Hard=(54.62, 44.07, 48.78)),
    "GTR-T5":                   dict(sa=69.45, Soft=(84.00, 66.76, 74.40), Hard=(54.62, 43.95, 48.71)),
    "MPNet (SBERT)":            dict(sa=61.05, Soft=(69.55, 56.71, 62.48), Hard=(50.47, 41.00, 45.25)),
    "SimCSE (BERT)":            dict(sa=72.67, Soft=(83.77, 66.07, 73.88), Hard=(54.62, 43.86, 48.65)),
    "Embedding-based (MiniLM)": dict(sa=62.15, Soft=(70.50, 57.43, 63.30), Hard=(51.18, 41.62, 45.91)),
    "Instruction-Tuned (E5)":   dict(sa=77.49, Soft=(97.39, 72.81, 83.32), Hard=(54.62, 43.90, 48.68)),
}
PAPER_STRICT = (10.55, 8.90, 9.65)

res = json.load(open(sys.argv[1]))
st = strict(load("data/HumanAnnotations.csv"), load("data/ChatGPTAnnotations.csv"))["PRF"]

rows, bad = [], 0
def cmp(name, what, got, want):
    global bad
    ok = abs(got - want) <= 0.01
    bad += not ok
    rows.append((name, what, want, got, "OK" if ok else "DIFF"))

for m, w in PAPER.items():
    r = res[m]
    cmp(m, "SAFE standalone", r["standalone"], w["sa"])
    for pen in ("Soft", "Hard"):
        for k, got, want in zip(("P", "R", "F1"), r[pen]["PRF"], w[pen]):
            cmp(m, f"{pen} {k}", got, want)
for k, got, want in zip(("P", "R", "F1"), st, PAPER_STRICT):
    cmp("(all models)", f"Strict {k}", got, want)

table = pd.DataFrame(rows, columns=["Model", "Metric", "Paper", "Reproduced", "Status"])
print(table.to_string(index=False))
print(f"\n{len(rows) - bad}/{len(rows)} numbers match the paper (tolerance 0.01).")
table.to_csv(sys.argv[2].rsplit(".", 1)[0] + "_check.csv", index=False)

# Redraw Fig. 2 from the reproduced numbers
order_sa = ["Embedding-based (MiniLM)", "GTR-T5", "Instruction-Tuned (E5)", "MPNet (SBERT)", "Sentence-T5", "SimCSE (BERT)"]
order = list(PAPER)
def prf_df(pen):
    return pd.DataFrame([[res[m][pen]["PRF"][2], res[m][pen]["PRF"][0], res[m][pen]["PRF"][1]] for m in order],
                        index=order, columns=["F1 Score (%)", "Precision (%)", "Recall (%)"])
strict_df = pd.DataFrame([[st[2], st[0], st[1]]] * 6, index=order, columns=["F1 Score (%)", "Precision (%)", "Recall (%)"])
fig, ax = plt.subplots(2, 2, figsize=(20, 15))
panels = [
    (pd.DataFrame({"SAFE Similarity Score (%)": [res[m]["standalone"] for m in order_sa]}, index=order_sa), "SAFE Similarity Score (%)", "YlGnBu"),
    (strict_df, "Strict Penalty - F1, Precision, Recall", "YlGnBu"),
    (prf_df("Soft"), "Soft Penalty - F1, Precision, Recall", "YlGnBu"),
    (prf_df("Hard"), "Hard Penalty - F1, Precision, Recall", "YlGnBu"),
]
for a, (df, title, cmap) in zip(ax.flat, panels):
    sns.heatmap(df, annot=True, fmt=".2f", cmap=cmap, vmin=0, vmax=100, ax=a)
    a.set_title(title, fontsize=16)
    a.set_ylabel("Model")
plt.tight_layout()
plt.savefig(sys.argv[2], dpi=150)
print("Saved", sys.argv[2])
sys.exit(1 if bad else 0)
