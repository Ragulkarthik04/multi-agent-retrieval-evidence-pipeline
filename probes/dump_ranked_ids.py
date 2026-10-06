
import argparse, csv, json, os, sys

def load_submission(submit_csv):
    q2supports = {}
    with open(submit_csv, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            q = r["question"]
            supports = json.loads(r["supports"])
            # keep order as the ranking
            q2supports[q] = supports
    return q2supports

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True, help="path to submission CSV (with 'question' and 'supports')")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--k", type=int, default=100, help="truncate ranking at top-k")
    args = ap.parse_args()

    q2supports = load_submission(args.submission)
    packed = [{"question": q, "ranked_ids": ids[:args.k]} for q, ids in q2supports.items()]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(packed, f, ensure_ascii=False, indent=2)

    print(f"✅ wrote ranked dump: {args.out} with {len(packed)} items (k={args.k})")

if __name__ == "__main__":
    main()
