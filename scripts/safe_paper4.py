"""Re-run Draft 14 cell 77 (Hard/Soft F1) + cell 79 (standalone SAFE) for all six models.

Scoring functions are copied verbatim from the notebook. Only changes:
Drive paths -> local args, and model.encode is memoised (same input -> same vector).
"""
import sys, json, warnings
import pandas as pd
from sentence_transformers import SentenceTransformer, util
warnings.filterwarnings("ignore")

human_file_path, chatgpt_file_path, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
only = sys.argv[4].split(",") if len(sys.argv) > 4 else None
SOFT_PENALTY_FACTOR = 0.50
MATCH_THRESHOLD = 0.80
PENALTY_OPTIONS = ["Hard", "Soft"]
MODEL_OPTIONS = {
    "1": ("Sentence-T5", "sentence-transformers/sentence-t5-base"),
    "2": ("GTR-T5", "sentence-transformers/gtr-t5-base"),
    "3": ("MPNet (SBERT)", "all-mpnet-base-v2"),
    "4": ("SimCSE (BERT)", "princeton-nlp/sup-simcse-bert-base-uncased"),
    "5": ("Embedding-based (MiniLM)", "all-MiniLM-L6-v2"),
    "6": ("Instruction-Tuned (E5)", "intfloat/e5-base"),
}

class Cached:
    def __init__(self, m): self.m, self.c = m, {}
    def encode(self, x, convert_to_tensor=True):
        if x not in self.c: self.c[x] = self.m.encode(x, convert_to_tensor=True)
        return self.c[x]

# ---- verbatim from cell 77 ----
def compute_similarity_qual(model, x, y):
    if x.strip().lower() == y.strip().lower():
        return 1.0
    x_emb = model.encode(x, convert_to_tensor=True)
    y_emb = model.encode(y, convert_to_tensor=True)
    return util.cos_sim(x_emb, y_emb).item()

def quadruple_similarity_qual(model, actual_quad, predicted_quad):
    if not actual_quad or not predicted_quad:
        return 0.0
    scores = []
    for actual_element, predicted_element in zip(actual_quad, predicted_quad):
        sim = compute_similarity_qual(model, actual_element, predicted_element)
        scores.append(sim)
    return sum(scores) / len(scores) * 100

def match_quads_qual(model, actual_quads, predicted_quads):
    total_score = 0
    matched = 0
    for pred_quad in predicted_quads:
        best_score = 0
        for act_quad in actual_quads:
            score = quadruple_similarity_qual(model, act_quad, pred_quad)
            if score > best_score:
                best_score = score
        total_score += best_score
        matched += 1
    if matched == 0:
        return 0.0
    return total_score / matched

def evaluate_reviews_qual(model, review_dict, threshold=MATCH_THRESHOLD):
    results = {}
    total_score = 0
    total_reviews = 0
    for review_id, quads in review_dict.items():
        score = match_quads_qual(model, quads.get("actual", []), quads.get("predicted", []))
        results[review_id] = round(score, 2)
        total_score += score
        total_reviews += 1
    results["overall_performance"] = round(total_score / total_reviews if total_reviews > 0 else 0.0, 2)
    return results

def compute_similarity(model, x, y):
    if x.strip().lower() == y.strip().lower():
        return 1.0
    x_emb = model.encode(x, convert_to_tensor=True)
    y_emb = model.encode(y, convert_to_tensor=True)
    sim = util.cos_sim(x_emb, y_emb).item()
    return (sim + 1) / 2

def quadruple_similarity_reweighted(model, penalty_type, actual_quad, predicted_quad):
    if not actual_quad or not predicted_quad:
        return 0.0
    aspect_similarity = compute_similarity(model, actual_quad[0], predicted_quad[0])
    opinion_similarity = compute_similarity(model, actual_quad[2], predicted_quad[2])
    category_match = actual_quad[1].strip().lower() == predicted_quad[1].strip().lower()
    category_score = 1.0 if category_match else (0.0 if penalty_type == "Hard" else SOFT_PENALTY_FACTOR)
    sentiment_match = actual_quad[3].strip().lower() == predicted_quad[3].strip().lower()
    sentiment_score = 1.0 if sentiment_match else (0.0 if penalty_type == "Hard" else SOFT_PENALTY_FACTOR)
    final_score = (aspect_similarity + opinion_similarity + category_score + sentiment_score) * 25
    return final_score / 100

def calculate_tp_fp_fn(actual_quads, predicted_quads, model, penalty_type, threshold=MATCH_THRESHOLD):
    matched_actual_indices, matched_predicted_indices, tp_pairs = set(), set(), []
    for p_idx, pred_quad in enumerate(predicted_quads):
        best_score, best_a_idx = 0, None
        for a_idx, act_quad in enumerate(actual_quads):
            score = quadruple_similarity_reweighted(model, penalty_type, act_quad, pred_quad)
            if score > best_score:
                best_score, best_a_idx = score, a_idx
        if best_score >= threshold:
            matched_actual_indices.add(best_a_idx)
            matched_predicted_indices.add(p_idx)
            tp_pairs.append((actual_quads[best_a_idx], pred_quad))
    FP = sum(1 for p in range(len(predicted_quads)) if p not in matched_predicted_indices)
    FN = sum(1 for a in range(len(actual_quads)) if a not in matched_actual_indices)
    return len(tp_pairs), FP, FN, len(matched_actual_indices)

def prf(TP, FP, FN):
    p = TP / (TP + FP) if TP + FP else 0.0
    r = TP / (TP + FN) if TP + FN else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(p * 100, 2), round(r * 100, 2), round(f * 100, 2)
# -------------------------------

def load_f1_inputs():  # as calculate_total_metrics: lower(), str(), strip() -- NaN becomes 'nan'
    h = pd.read_csv(human_file_path).map(lambda x: x.lower() if isinstance(x, str) else x)
    c = pd.read_csv(chatgpt_file_path).map(lambda x: x.lower() if isinstance(x, str) else x)
    h["Review ID"] = h["Review ID"].astype(str); c["Review ID"] = c["Review ID"].astype(str)
    q = lambda df: [[str(r[k]).strip() for k in ("Aspect", "Category", "Opinion", "Sentiment")] for _, r in df.iterrows()]
    ids = set(h["Review ID"]) | set(c["Review ID"])
    return {i: (q(h[h["Review ID"] == i]), q(c[c["Review ID"] == i])) for i in ids}

def load_standalone_inputs():  # as cell 79: lower().strip(), dropna() per quad
    h = pd.read_csv(human_file_path).map(lambda x: x.lower().strip() if isinstance(x, str) else x)
    c = pd.read_csv(chatgpt_file_path).map(lambda x: x.lower().strip() if isinstance(x, str) else x)
    h["Review ID"] = h["Review ID"].astype(str); c["Review ID"] = c["Review ID"].astype(str)
    cols = ["Aspect", "Category", "Opinion", "Sentiment"]
    ids = set(h["Review ID"]) | set(c["Review ID"])
    return {i: {"actual": h[h["Review ID"] == i][cols].dropna().values.tolist(),
                "predicted": c[c["Review ID"] == i][cols].dropna().values.tolist()} for i in ids}

f1_in, sa_in = load_f1_inputs(), load_standalone_inputs()
out = {}
for mid, (name, path) in MODEL_OPTIONS.items():
    if only and mid not in only: continue
    model = Cached(SentenceTransformer(path))
    res = {"standalone": evaluate_reviews_qual(model, sa_in)["overall_performance"]}
    for pen in PENALTY_OPTIONS:
        TP = FP = FN = M = 0
        for a, p in f1_in.values():
            t, f, n, m = calculate_tp_fp_fn(a, p, model, pen)
            TP += t; FP += f; FN += n; M += m
        res[pen] = dict(TP=TP, FP=FP, FN=FN, distinct_gold_matched=M, PRF=prf(TP, FP, FN))
    out[name] = res
    print(name, json.dumps(res), flush=True)
    json.dump(out, open(out_json, "w"), indent=2)
