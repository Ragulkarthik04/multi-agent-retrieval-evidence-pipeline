
"""
evaluate_map.py — Single-metric MAP scorer (options only)

Usage:
  python evaluate_map.py <submission_csv> <gold_json> [--debug N]

- submission_csv columns: question, distribution (JSON), supports (ignored)
- gold_json formats supported:
    1) {"questions": [{"question": "...", "distribution": {...}} , ...]}
    2) [{"question": "...", "distribution": {...}}, ...]
    3) {"<question text>": {"distribution": {...}}, ...}

We normalise question text and option strings, parse numeric strings like "20%"
and apply a small alias map (e.g., "not at all" <-> "nothing at all").
"""

import csv
import json
import re
import sys
import argparse
from collections import OrderedDict

# helpers 

def norm_text(s: str) -> str:
    """Lowercase, strip, collapse spaces, drop simple punctuation."""
    s = (s or "").strip().lower()

    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")

    s = re.sub(r"[^a-z0-9\s/]", " ", s)  
    s = re.sub(r"\s+", " ", s).strip()
    return s


OPTION_ALIASES = {
    "nothing at all": "not at all",
    "not at all": "not at all",
    # spirituality long labels → short
    "had a sudden feeling of connection with something from beyond this world": "connection feeling",
    "had a strong feeling that someone who has passed away was communicating with you from beyond this world": "felt communication from deceased",
    "believe that spirits or unseen spiritual forces exist and have personally encountered one": "encountered spirit",
    # short → short (idempotent)
    "connection feeling": "connection feeling",
    "felt communication from deceased": "felt communication from deceased",
    "encountered spirit": "encountered spirit",
    # common survey options
    "a great deal": "a great deal",
    "a fair amount": "a fair amount",
    "some": "some",
    "not too much": "not too much",
    "yes": "yes",
    "no": "no",
    "making things better": "making things better",
    "making things worse": "making things worse",
    "not having much effect": "not having much effect",
}

def alias_opt(opt: str) -> str:
    k = norm_text(opt)
    return OPTION_ALIASES.get(k, k)

def safe_float(x):
    """Parse floats from strings robustly (handles '20%', '20', '0.2')."""
    try:
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        if not m:
            return 0.0
        val = float(m.group(0))
        if "%" in s and val > 1.0:
            val = val / 100.0
        return val
    except Exception:
        return 0.0

def load_gold(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "questions" in data:
        items = data["questions"]
    elif isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        # mapping question -> obj
        items = [{"question": k, **(v if isinstance(v, dict) else {})} for k, v in data.items()]
    else:
        raise ValueError("Unsupported gold JSON format")

    gold = {}
    for it in items:
        q = norm_text(it.get("question", ""))
        dist = it.get("distribution", {})
        # normalise option keys & parse numeric
        gmap = {}
        for k, v in (dist or {}).items():
            ak = alias_opt(k)
            gmap[ak] = gmap.get(ak, 0.0) + safe_float(v)
        # re-normalise 
        s = sum(gmap.values())
        if s > 0:
            for k in list(gmap.keys()):
                gmap[k] = gmap[k] / s
        gold[q] = gmap
    return gold

def load_pred_csv(path):
    preds = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            q = norm_text(r["question"])
            dist_s = r.get("distribution", "{}")
            try:
                dist = json.loads(dist_s)
            except Exception:
                dist = {}
            pmap = {}
            for k, v in (dist or {}).items():
                ak = alias_opt(k)
                pmap[ak] = pmap.get(ak, 0.0) + safe_float(v)
            # normalise
            s = sum(pmap.values())
            if s > 0:
                for k in list(pmap.keys()):
                    pmap[k] = pmap[k] / s
            preds[q] = pmap
    return preds

def average_precision(rank_list, relevant_set):
    """
    rank_list: list of option strings ranked high→low
    relevant_set: set of relevant options (gold prob > 0)
    """
    if not rank_list or not relevant_set:
        return 0.0
    hit = 0
    precisions = []
    for i, opt in enumerate(rank_list, 1):
        if opt in relevant_set:
            hit += 1
            precisions.append(hit / i)
    if not precisions:
        return 0.0
    return sum(precisions) / len(relevant_set)

# main 

def main():
    ap = argparse.ArgumentParser(description="Compute MAP (options only) for MAS A2.")
    ap.add_argument("submission_csv")
    ap.add_argument("gold_json")
    ap.add_argument("--debug", type=int, default=0, help="print overlap for first N questions")
    args = ap.parse_args()

    gold = load_gold(args.gold_json)
    preds = load_pred_csv(args.submission_csv)

    # intersect by normalised question text
    common = [q for q in preds.keys() if q in gold]
    if not common:
        
        print("\n⚠️ No overlapping questions. Check normalisation.\n")
        print("📋 sample in CSV:")
        for q in list(preds.keys())[:3]:
            print(" •", q[:120])
        print("\n📋 sample in gold:")
        for q in list(gold.keys())[:3]:
            print(" •", q[:120])
        sys.exit(1)

    if args.debug > 0:
        print("\n🔍 Overlap debug (first {} questions)\n".format(min(args.debug, len(common))))
        for q in common[:args.debug]:
            pmap = preds[q]
            gmap = gold[q]
            pred_keys = [k for k, _ in sorted(pmap.items(), key=lambda kv: (-kv[1], kv[0]))]
            gold_keys = sorted(gmap.keys())
            overlap = [k for k in pred_keys if k in gold_keys]
            print(f"Q: {q[:100]} …")
            print("  pred keys:", pred_keys)
            print("  gold keys:", gold_keys)
            print("  overlap  :", overlap)
            print()

    # compute MAP
    APs = []
    for q in common:
        pmap = preds[q]
        gmap = gold[q]
        
        relevant = {k for k, v in gmap.items() if v > 0.0}
        ranking = [k for k, _ in sorted(pmap.items(), key=lambda kv: (-kv[1], kv[0]))]
        ap = average_precision(ranking, relevant)
        APs.append(ap)

    mean_ap = sum(APs) / len(APs) if APs else 0.0
    print(f"\n✅ Evaluated {len(common)} questions")
    print(f"📊 Mean AP: {mean_ap:.6f}")

if __name__ == "__main__":
    main()
