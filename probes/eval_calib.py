
import argparse, csv, json, math, sys
from statistics import mean

def load_subset(path):
    with open(path, "r", encoding="utf-8") as f:
        ids = json.load(f)
    if not isinstance(ids, list):
        raise ValueError("subset file must be a JSON list of question strings")
    return set(ids)

def load_predictions(path):
    preds = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            q = r["question"]
            dist = json.loads(r["distribution"])
            preds[q] = {k: float(v) for k, v in dist.items()}
    return preds

def load_truth(path):
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    
    if isinstance(j, dict):
        if "data" in j: j = j["data"]
        elif "questions" in j: j = j["questions"]

    truth = {}
    if isinstance(j, list):
        for item in j:
            
            if isinstance(item, dict) and "question" in item and "distribution" in item:
                truth[item["question"]] = {k: float(v) for k, v in item["distribution"].items()}
           
            elif isinstance(item, str):
                continue
   
    return truth

def entropy(pdict):
    vals = [max(float(v), 0.0) for v in pdict.values()]
    s = sum(vals)
    if s <= 0:
        return 0.0
    vals = [v / s for v in vals]
    return -sum(0.0 if v <= 0 else v * math.log(v) for v in vals)

def js_divergence(p, q, eps=1e-12):
    
    keys = set(p) | set(q)
    pv = []
    qv = []
    for k in keys:
        pv.append(max(float(p.get(k, 0.0)), 0.0))
        qv.append(max(float(q.get(k, 0.0)), 0.0))
    sp = sum(pv)
    sq = sum(qv)
    if sp <= 0 or sq <= 0:
        return 0.0
    pv = [x / sp for x in pv]
    qv = [x / sq for x in qv]
    m = [(a + b) / 2.0 for a, b in zip(pv, qv)]
    def kl(a, b):
        s = 0.0
        for ai, bi in zip(a, b):
            if ai > 0:
                s += ai * math.log((ai + eps) / (bi + eps))
        return s
    return 0.5 * kl(pv, m) + 0.5 * kl(qv, m)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True, help="JSON list of 15 probe question strings")
    ap.add_argument("--pred", required=True, help="submission CSV")
    ap.add_argument("--truth", default=None, help="(optional) dev_groundtruth.json with distributions")
    args = ap.parse_args()

    subset = load_subset(args.subset)
    preds = load_predictions(args.pred)
    truth = load_truth(args.truth)

    entropies = []
    maxprobs = []
    js_scores = []

    missing = 0
    for q in subset:
        pd = preds.get(q)
        if not pd:
            missing += 1
            continue
        # intrinsic calibration stats
        entropies.append(entropy(pd))
        maxprobs.append(max(pd.values()) if pd else 0.0)
        # optional JS if truth present for this question
        if truth and q in truth:
            js_scores.append(js_divergence(pd, truth[q]))

    print(f"✅ evaluated {len(subset) - missing} probe questions (missing in pred: {missing})")
    if entropies:
        print("Mean entropy (nats):", round(mean(entropies), 3))
        print("Mean max-prob      :", round(mean(maxprobs), 3))
    else:
        print("No overlapping questions between subset and predictions.")

    if args.truth:
        if js_scores:
            print("Mean JS divergence :", round(mean(js_scores), 6))
        else:
            print("Truth loaded but contains no per-question distributions (or no overlaps) — JS skipped.")

if __name__ == "__main__":
    main()
