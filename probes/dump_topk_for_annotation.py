
import argparse, csv, json, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=30, help="how many IDs to show per question")
    args = ap.parse_args()

    rows = []
    with open(args.submission, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            q = r["question"]
            ids = json.loads(r["supports"])[:args.k]
            rows.append({
                "question": q,
                "topk_candidate_ids": json.dumps(ids, ensure_ascii=False),
                "relevant_doc_ids_FILL_THIS": "[]"  
            })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"📝 wrote: {args.out} — fill 'relevant_doc_ids_FILL_THIS' with true positives")

if __name__ == "__main__":
    main()
