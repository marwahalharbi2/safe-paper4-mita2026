# SAFE — Paper 4 reproduction (MITA 2026)

[![DOI](https://zenodo.org/badge/1425440333.svg)](https://doi.org/10.5281/zenodo.23486470)
[![Paper](https://img.shields.io/badge/IEEE-10.1109%2FMITA69365.2026.11581671-blue)](https://doi.org/10.1109/MITA69365.2026.11581671)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](LICENSE)

Code and data that reproduce every number in Fig. 2 of:

> M. Alharbi, J. Yin, and Y. Miao, "SAFE: A Semantic-Aware Flexible Evaluation Framework for Large Language Models-Based Sentiment Analysis," in *2026 13th International Conference on Machine Intelligence Theory and Applications (MiTA)*, IEEE, 2026, pp. 491–498. doi: [10.1109/MITA69365.2026.11581671](https://doi.org/10.1109/MITA69365.2026.11581671)

## How to cite

Please cite the paper, and the software archive if you use the code or data:

```bibtex
@inproceedings{alharbi2026safe,
  author    = {Alharbi, Marwah and Yin, Jiao and Miao, Yuan},
  title     = {{SAFE}: A Semantic-Aware Flexible Evaluation Framework for Large Language Models-Based Sentiment Analysis},
  booktitle = {2026 13th International Conference on Machine Intelligence Theory and Applications (MiTA)},
  publisher = {IEEE},
  year      = {2026},
  pages     = {491--498},
  doi       = {10.1109/MITA69365.2026.11581671}
}

@software{alharbi2026safe_code,
  author    = {Alharbi, Marwah},
  title     = {{SAFE}: Reproduction Code and Data for "A Semantic-Aware Flexible Evaluation Framework for Large Language Models-Based Sentiment Analysis" (MITA 2026)},
  version   = {v1.0},
  publisher = {Zenodo},
  year      = {2026},
  doi       = {10.5281/zenodo.23486470}
}
```

GitHub's "Cite this repository" button (from `CITATION.cff`) gives the same information.

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

## Equations as published

Notation: a quadruple is (aspect *a*, opinion *o*, category *c*, sentiment *s*). $q = (y_a, y_o, y_c, y_s)$ is a reference quadruple in the reference set $Q$, and $\hat q = (\hat y_a, \hat y_o, \hat y_c, \hat y_s)$ is a predicted quadruple in the prediction set $\hat Q$.

**SAFE standalone**

$$\mathrm{SAFE}_{\mathrm{Standalone}}(\hat q, q) = \mathrm{Sim}(\hat q, q)$$

**Hard SAFE and Soft SAFE**

$$\mathrm{SAFE}_{\mathrm{Hard}}(\hat q, q) = \lambda_1 \mathrm{Sim}(y_a, \hat y_a) + \lambda_2 \mathrm{Sim}(y_o, \hat y_o) + \lambda_3 \mathrm{Exm}(y_c, \hat y_c) + \lambda_4 \mathrm{Exm}(y_s, \hat y_s)$$

$$\mathrm{SAFE}_{\mathrm{Soft}}(\hat q, q) = \lambda_1 \mathrm{Sim}(y_a, \hat y_a) + \lambda_2 \mathrm{Sim}(y_o, \hat y_o) + \lambda_3 \mathrm{Sim}(y_c, \hat y_c) + \lambda_4 \mathrm{Sim}(y_s, \hat y_s)$$

where $\mathrm{Exm}(y, \hat y) = 1$ if $y = \hat y$ and 0 otherwise, Sim is "cosine similarity between embedding vectors", and $\lambda_1 + \dots + \lambda_4 = 1$.

**TP / FP / FN** (SAFE in either mode)

$$\mathrm{TP}(\hat q, q) = \begin{cases} 1 & \text{if } \hat q \in (\hat Q \cap Q) \text{ and } \mathrm{SAFE}(\hat q, q) \ge \tau \\ 0 & \text{otherwise} \end{cases}$$

$$\mathrm{FP}(\hat q, q) = \begin{cases} 1 & \text{if } \hat q \in (\hat Q \setminus Q) \text{ and } \mathrm{SAFE}(\hat q, q) < \tau \\ 0 & \text{otherwise} \end{cases}$$

$$\mathrm{FN}(\hat q, q) = \begin{cases} 1 & \text{if } q \in (Q \setminus \hat Q) \text{ and } \mathrm{SAFE}(\hat q, q) < \tau \\ 0 & \text{otherwise} \end{cases}$$

**Precision, Recall, F1**

$$\mathrm{Precision}_{\mathrm{SAFE}} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FP}} \qquad \mathrm{Recall}_{\mathrm{SAFE}} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FN}} \qquad \mathrm{F1}_{\mathrm{SAFE}} = \frac{2 \cdot \mathrm{Precision}_{\mathrm{SAFE}} \cdot \mathrm{Recall}_{\mathrm{SAFE}}}{\mathrm{Precision}_{\mathrm{SAFE}} + \mathrm{Recall}_{\mathrm{SAFE}}}$$

The next section gives the equations as implemented. The differences are summarised under [Published description vs. implementation](#published-description-vs-implementation).

## Equations as implemented

$$\mathrm{Sim}(y, \hat y) = \begin{cases} 1 & \text{if } y = \hat y \text{ (after lower-casing and trimming)} \\ \dfrac{\cos(\mathbf e_y, \mathbf e_{\hat y}) + 1}{2} & \text{otherwise} \end{cases}$$

$$\mathrm{Pen}_\rho(y, \hat y) = \begin{cases} 1 & \text{if } y = \hat y \\ \rho & \text{otherwise} \end{cases}$$

$$\mathrm{SAFE}_{\mathrm{mode}}(\hat q, q) = \lambda_1 \mathrm{Sim}(y_a, \hat y_a) + \lambda_2 \mathrm{Sim}(y_o, \hat y_o) + \lambda_3 \mathrm{Pen}_\rho(y_c, \hat y_c) + \lambda_4 \mathrm{Pen}_\rho(y_s, \hat y_s)$$

with $\rho = 0$ for Hard SAFE (identical to the published Hard equation, since $\mathrm{Pen}_0 = \mathrm{Exm}$), $\rho = 0.5$ for Soft SAFE, $\lambda_1 = \dots = \lambda_4 = 0.25$, and $\tau = 0.80$.

## Why Hard and Soft are designed this way

Aspect and opinion are free-text spans. Their boundaries are subjective (annotators disagree on them), so they are always compared by embedding similarity. Category and sentiment are labels from a fixed set, and embedding similarity between label names says little: under the rescaled scale, *positive* vs *negative* can still score about 0.8. So a mismatched label gets a fixed credit ρ instead. Hard SAFE (ρ = 0) is meant for precision-critical uses, where a wrong label should count as a failure. Soft SAFE (ρ = 0.5) is meant for evaluating generative models, where near-misses should still earn credit.

With τ = 0.80 and equal weights, this gives each mode a simple decision rule:

| Labels wrong | Hard: max score | Soft: max score | Can it be a TP (≥ 0.80)? |
|---|---|---|---|
| none | 1.00 | 1.00 | both modes: yes, if aspect and opinion are close enough |
| one | 0.75 | 0.875 | **Hard: never.** Soft: yes, if aspect and opinion have mean similarity ≥ 0.85 |
| both | 0.50 | 0.75 | never |

- **Hard SAFE** = labels must match exactly; only the aspect and opinion wording may vary.
- **Soft SAFE** = additionally forgives **one** label error, when the spans are nearly identical.

This is why Hard SAFE precision is identical (54.62%) for four of the six models: the label check decides almost every match, so the embedding model hardly matters. Under Soft SAFE the embedding model decides which spans count as close, so the models separate (F1 62.48–83.49).

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

## Published description vs. implementation

The code is the version that produced the published figure, so it is authoritative for the reported numbers. **No number or figure differs.** The table maps each point of the published method description to what the code does:

| # | Published paper | Implementation |
|---|---|---|
| 1 | Soft SAFE compares category and sentiment with Sim | A mismatched category or sentiment gets a fixed partial credit ρ = 0.5 (`Pen_ρ`) |
| 2 | Sim is "cosine similarity" | Sim = 1 for identical strings, otherwise (cos + 1) / 2. The standalone score uses raw cosine |
| 3 | TP/FP/FN written as set membership | Best match per prediction within a review, not one-to-one, so TP + FN can exceed the number of gold quads |
| 4 | Empty (implicit) elements not mentioned | Treated as the token "nan" in the F1 modes; skipped in the standalone score |
| 5 | τ, λ not stated | τ = 0.80, λ = 0.25 each, ρ = 0.5 |
| 6 | "Five" embedding models | Six: Sentence-T5 appears in Fig. 2 but not in the paper's list |

## Environment used for this reproduction

Python 3.11.9, torch 2.14.1 (CPU), sentence-transformers 6.1.0, transformers 5.20.1, pandas 3.0.6, Windows 11.

The original runs (paper) used Google Colab with a T4 GPU, Python 3.11.13, PyTorch 2.6.0, Transformers 4.52.4. The reproduced numbers agree to two decimals in both environments.

## Related

- Dataset source: https://github.com/marwahalharbi2/QuadABSA-TourismDataset
- Human-evaluation study of SAFE (Paper 5): https://github.com/marwahalharbi2/safe-quad-absa
