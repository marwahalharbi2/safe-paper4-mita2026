"""Strict (exact-match) F1 for Paper 4, Fig. 2 top-right panel.

A predicted quad is a TP only if all four elements equal a not-yet-used gold
quad in the same review (lower-cased, stripped). Embedding-free, so the result
is the same for every model row of the figure.

Usage: python scripts/strict_f1.py data/HumanAnnotations.csv data/ChatGPTAnnotations.csv
"""
import sys
import pandas as pd

COLS = ["Aspect", "Category", "Opinion", "Sentiment"]


def load(path):
    df = pd.read_csv(path).map(lambda x: x.lower() if isinstance(x, str) else x)
    df["Review ID"] = df["Review ID"].astype(str)
    return df


def quads(df):
    return [tuple(str(r[k]).strip() for k in COLS) for _, r in df.iterrows()]


def strict(human, chatgpt):
    tp = fp = fn = 0
    for rid in set(human["Review ID"]) | set(chatgpt["Review ID"]):
        gold = quads(human[human["Review ID"] == rid])
        pred = quads(chatgpt[chatgpt["Review ID"] == rid])
        used = set()
        for p in pred:
            j = next((j for j, g in enumerate(gold) if g == p and j not in used), None)
            if j is None:
                fp += 1
            else:
                tp += 1
                used.add(j)
        fn += len(gold) - len(used)
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return dict(TP=tp, FP=fp, FN=fn, PRF=(round(p * 100, 2), round(r * 100, 2), round(2 * p * r / (p + r) * 100, 2)))


if __name__ == "__main__":
    print(strict(load(sys.argv[1]), load(sys.argv[2])))
