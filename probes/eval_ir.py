
import argparse, json, math, statistics as st

def ndcg_at_k(rels, k):
    
    rels_k = rels[:k]
    dcg = sum((rel / math.log2(i+2)) for i, rel in enumerate(rels_k))
    ideal = sorted(rels, reverse=True)[:k]
    idcg = sum((rel / math.log2(i+2)) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0

def recall_at_k(rels, num_rel, k):
    return min(sum(rels[:k]), num_rel) / (num_rel if num_rel > 0 else 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True, help="probes/ir_probe.json")
    ap.add_argument("--ranked", required=True, help="ranked dump JSON from dump_ranked_ids.py")
    ap.add_argument("--k", nargs="+", type=int, default=[10,50,100])
    args = ap.parse_args()

    probe = {item["question"]: set(item["relevant_doc_ids"]) for item in json.load(open(args.probe, encoding="utf-8"))}
    ranked = {item["question"]: item["ranked_ids"] for item in json.load(open(args.ranked, encoding="utf-8"))}

    ks = sorted(set(args.k))
    rec = {k: [] for k in ks}
    ndc = {k: [] for k in ks}
    evaluated = 0

    for q, relset in probe.items():
        if q not in ranked:
            continue
        ranking = ranked[q]
        rels = [1 if d in relset else 0 for d in ranking]
        num_rel = len(relset)
        for k in ks:
            rec[k].append(recall_at_k(rels, num_rel, k))
            ndc[k].append(ndcg_at_k(rels, k))
        evaluated += 1

    if evaluated == 0:
        print("No overlap between probe questions and ranked dump."); return

    print(f"✅ evaluated {evaluated} probe questions")
    for k in ks:
        print(f"R@{k}: {st.mean(rec[k]):.3f} | nDCG@{k}: {st.mean(ndc[k]):.3f}")

if __name__ == "__main__":
    main()
