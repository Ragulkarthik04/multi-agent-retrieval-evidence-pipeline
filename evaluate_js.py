
import json, csv, sys, re
import numpy as np
from scipy.spatial.distance import jensenshannon

def norm_question(s: str) -> str:
    s = s.lower().strip()
    s = s.replace("?", "")
    s = s.replace('"', "")
    s = re.sub(r"\s+", " ", s)
    return s

def norm_option(s: str) -> str:
    s = s.lower().strip()
    
    s = re.sub(r"[^\w\s%&/\-']", " ", s)  
    s = re.sub(r"\s+", " ", s)
    return s

def read_preds(csv_path):
    preds = {}
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            q = norm_question(r["question"])
            try:
                d = json.loads(r["distribution"])
            except Exception:
                
                d = json.loads(json.loads(r["distribution"]))
            
            preds[q] = {norm_option(k): float(v) for k, v in d.items()}
    return preds

def read_truths(dev_json_path):
    raw = json.load(open(dev_json_path, encoding="utf-8"))
    truths = {}
    
    if isinstance(raw, list):
        for it in raw:
            if isinstance(it, dict) and "question" in it and "distribution" in it:
                q = norm_question(it["question"])
                truths[q] = {norm_option(k): float(v) for k, v in it["distribution"].items()}
    elif isinstance(raw, dict):
        for qtext, obj in raw.items():
            if isinstance(obj, dict) and "distribution" in obj:
                q = norm_question(qtext)
                truths[q] = {norm_option(k): float(v) for k, v in obj["distribution"].items()}
    return truths

def align_vectors(pdist, gdist):
    """Return (p_vec, g_vec, keys_used, unmapped_pred, unmapped_gold)"""
    
    p_keys = set(pdist.keys())
    g_keys = set(gdist.keys())
    inter = sorted(p_keys & g_keys)
    unmapped_pred = sorted(p_keys - g_keys)
    unmapped_gold = sorted(g_keys - p_keys)

    if not inter:
        return None, None, [], unmapped_pred, unmapped_gold

    p_vec = np.array([pdist.get(k, 0.0) for k in inter], dtype=float)
    g_vec = np.array([gdist.get(k, 0.0) for k in inter], dtype=float)

    
    def renorm(v):
        s = float(v.sum())
        if s <= 0:
            return np.ones_like(v) / len(v)
        out = v / s
        
        return out / float(out.sum())

    return renorm(p_vec), renorm(g_vec), inter, unmapped_pred, unmapped_gold

def main(pred_csv, dev_json):
    preds = read_preds(pred_csv)
    truths = read_truths(dev_json)

   
    print("📋 sample in CSV:", *(list(preds.keys())[:3]), sep="\n • ", end="\n\n")
    print("📋 sample in dev:", *(list(truths.keys())[:3]), sep="\n • ", end="\n\n")

    scores = []
    mismatches = 0
    detailed = []

    
    for tq, gdist in truths.items():
        pdist = preds.get(tq)
        if pdist is None:
            mismatches += 1
            detailed.append((tq, "no_match_in_csv", [], [], []))
            continue

        p_vec, g_vec, keys, up, ug = align_vectors(pdist, gdist)
        if p_vec is None:
            mismatches += 1
            detailed.append((tq, "no_option_overlap", [], up, ug))
            continue

        js = jensenshannon(p_vec, g_vec, base=2)
        if np.isnan(js):
            mismatches += 1
            detailed.append((tq, "nan_js", keys, up, ug))
            continue

        scores.append(float(js))
        if up or ug:
            detailed.append((tq, "partial_option_overlap", keys, up, ug))

    if scores:
        print(f"✅ Evaluated {len(scores)} / {len(truths)} questions")
        print(f"📊 Mean JS divergence (lower better): {round(float(np.mean(scores)), 6)}")
    else:
        print("⚠️ Could not compute JS for any question.")

    if mismatches:
        print("\n🔎 Notes about mismatches / option alignment:")
        for q, kind, keys, up, ug in detailed[:10]:  
            print(f"— {kind}: {q[:80]}…")
            if up:
                print(f"   • in CSV only (unmapped): {up}")
            if ug:
                print(f"   • in dev only (unmapped): {ug}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python evaluate_js.py <pred_csv> <dev_json>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
