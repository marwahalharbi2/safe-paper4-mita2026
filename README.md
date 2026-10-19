# SAFE — Paper 4 reproduction (MITA 2026)

Code and data that reproduce every number in Fig. 2 of:

> M. Alharbi, J. Yin, Y. Miao. *SAFE: A Semantic-Aware Flexible Evaluation Framework for Large Language Models-Based Sentiment Analysis.* MITA 2026 (IEEE).

Fig. 2 reports four panels for six embedding models: the standalone SAFE similarity score, and Precision / Recall / F1 under Strict (exact match), Hard SAFE and Soft SAFE.

**Status: 45 / 45 published numbers reproduced exactly** (`results/fig2_reproduced_check.csv`).

## Quick start

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1. Hard/Soft SAFE F1 + standalone SAFE, all six models (~1-2 h on CPU, minutes on a GPU)
python scripts/safe_paper4.py data/HumanAnnotations.csv data/ChatGPTAnnotations.csv results/results.json

# 2. Strict exact-match F1 (seconds)
python scripts/strict_f1.py data/HumanAnnotations.csv data/ChatGPTAnnotations.csv

# 3. Compare against the paper and redraw Fig. 2 (exits non-zero on any mismatch)
python scripts/check_and_plot.py results/results.json results/fig2_reproduced.png
```

Run one model only by adding its ID (1–6) as a fourth argument to step 1, e.g. `... results/results.json 5` for MiniLM.

## Contents

| Path | What it is |
|---|---|
| `data/HumanAnnotations.csv` | Gold quadruples: 1,000 rows, 231 hotel reviews (`Review ID, Review, Aspect, Category, Opinion, Sentiment`) |
| `data/ChatGPTAnnotations.csv` | ChatGPT-predicted quadruples for the same reviews: 844 rows |
| `scripts/safe_paper4.py` | The scoring code of the original notebook, copied verbatim; paths made local, `model.encode` memoised |
| `scripts/strict_f1.py` | Exact-match F1 (all four elements identical, one-to-one within a review) |
| `scripts/check_and_plot.py` | Checks each reproduced number against the published value and redraws Fig. 2 |
| `results/` | Output of a full run: `results.json`, the check table, the redrawn figure |
| `original_notebook/Draft_14_Evaluation.ipynb` | The original Colab notebook with its saved outputs. The paper's numbers come from the section "Final Final Final Approach" (cells 77 and 79–80) |

Models (Hugging Face IDs): `sentence-transformers/sentence-t5-base`, `sentence-transformers/gtr-t5-base`, `all-mpnet-base-v2`, `princeton-nlp/sup-simcse-bert-base-uncased`, `all-MiniLM-L6-v2`, `intfloat/e5-base`.

Settings: match threshold τ = 0.80, element weights λ = 0.25 each, soft penalty factor 0.50.

## What the code computes

Text is lower-cased and stripped. Quadruples are grouped by `Review ID`.

**Element similarity (F1 modes).** 1.0 if the two strings are identical, otherwise `(cos + 1) / 2` of the two sentence embeddings.

**Quad score.** `0.25·Sim(aspect) + 0.25·Sim(opinion) + 0.25·c + 0.25·s`, where for category (c) and sentiment (s):
- Hard SAFE: 1 if exact match, else 0
- Soft SAFE: 1 if exact match, else 0.5

**TP / FP / FN.** Within a review, each prediction is compared with every gold quad. If its best score is ≥ 0.80 the prediction is a TP, otherwise an FP. Gold quads that are nobody's best match are FN.

**Standalone SAFE.** For each prediction, the best mean element similarity over the review's gold quads (raw cosine, no rescaling), averaged over predictions and then over all 231 reviews. Rows with an empty cell are dropped. A review with no gold quads scores 0.

**Strict.** A prediction is a TP only if all four elements equal an unused gold quad in the same review.

## Notes for anyone extending this work

The code reproduces the published figure. In four places it differs from how the paper describes the method. These are listed so results can be compared like with like:

1. **Soft SAFE.** The paper's Soft equation compares category and sentiment by embedding similarity. The code gives a flat 0.5 credit for any mismatch instead.
2. **Matching is not one-to-one.** Several predictions can claim the same gold quad. So TP + FN does not equal the number of gold quads: Soft SAFE with Sentence-T5 gives TP 822 + FN 303 = 1,125 against 1,000 gold quads, since only 697 distinct gold quads are matched. This raises recall. `results.json` records `distinct_gold_matched` for every run.
3. **Two similarity scales.** The paper writes Sim as cosine similarity. The F1 modes use `(cos + 1) / 2`, which puts unrelated strings near 0.5. The standalone score uses raw cosine.
4. **Empty cells.** In the F1 modes an empty cell becomes the string `"nan"`, so two empty cells count as an exact match. 124 gold aspects and 376 gold opinions are empty (implicit). The standalone score instead drops rows with an empty cell.

A corrected, one-to-one implementation is a separate piece of work and is not part of this repository.

## Corrections in the thesis version (Chapter 6)

The PhD thesis reprints this paper as Chapter 6. Its method text has been corrected so that it describes the code above. **No number or figure changed.** Differences from the published Version of Record:

| # | Published paper | Thesis Chapter 6 |
|---|---|---|
| 1 | Soft SAFE compares category and sentiment with Sim | Soft SAFE gives a mismatched category or sentiment a fixed partial credit ρ = 0.5 (`Pen_ρ`) |
| 2 | Sim is "cosine similarity" | Sim = 1 for identical strings, otherwise (cos + 1) / 2; the compressed scale is noted. The standalone score uses raw cosine |
| 3 | TP/FP/FN written as set membership | Matching described as it runs: best match per prediction within a review, not one-to-one, so TP + FN can exceed the number of gold quads |
| 4 | Empty (implicit) elements not mentioned | Stated: treated as the token "nan" in the F1 modes, skipped in the standalone score |
| 5 | τ, λ not stated | τ = 0.80, λ = 0.25 each, ρ = 0.5 stated |
| 6 | "Five" embedding models | Six: Sentence-T5 was in Fig. 2 but left out of the list |

## Environment used for this reproduction

Python 3.11.9, torch 2.14.1 (CPU), sentence-transformers 6.1.0, transformers 5.20.1, pandas 3.0.6, Windows 11.

The original runs (paper) used Google Colab with a T4 GPU, Python 3.11.13, PyTorch 2.6.0, Transformers 4.52.4. The reproduced numbers agree to two decimals in both environments.

## Related

- Dataset source: https://github.com/marwahalharbi2/QuadABSA-TourismDataset
- Human-evaluation study of SAFE (Paper 5): https://github.com/marwahalharbi2/safe-quad-absa
