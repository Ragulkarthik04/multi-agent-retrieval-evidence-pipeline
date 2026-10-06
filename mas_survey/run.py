

"""
MAS Runner — Retriever + Aggregator 
"""

import os
import sys
import csv
import json
import random
from typing import List, Tuple, Dict, Any
import numpy as np
import yaml

# Whoosh (sparse)
from whoosh import index as windex
from whoosh.qparser import MultifieldParser, OrGroup
from whoosh.scoring import BM25F

# Utilities

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

def ensure_dir(p: str):
    d = os.path.dirname(p) or "."
    if d and not os.path.exists(d):
        os.makedirs(d)

def softmax_temperature(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        temperature = 1.0
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x / float(temperature))
    s = e.sum()
    if s <= 0:
        return np.ones_like(x) / float(len(x))
    return e / s

def normalize_probs(p: np.ndarray) -> np.ndarray:
    p = np.maximum(p, 0.0)
    s = p.sum()
    if s <= 0:
        return np.ones_like(p) / float(len(p))
    p = p / s
    # enforce sum to 1 ± 1e-6
    if abs(p.sum() - 1.0) > 1e-6:
        p = p / p.sum()
    return p

def write_csv(path: str, rows: List[Tuple[str, Dict[str, float], List[str]]]):
    ensure_dir(path)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question", "distribution", "supports"])
        for qtext, dist_map, supports in rows:
            w.writerow([
                qtext,
                json.dumps(dist_map, ensure_ascii=True),
                json.dumps(supports, ensure_ascii=True),
            ])

# Input loader 

def load_questions(path: str) -> List[Dict[str, Any]]:
    """
    Accepts:
      - list of {"id","question","options"} or {"question","options"}
      - {"questions": [...]}
      - dict mapping question_text -> {"options":[...]} or {"distribution":{...}}
    Returns normalized list: [{"id","question","options"}]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "questions" in data:
        data = data["questions"]

    if isinstance(data, list):
        out = []
        for i, q in enumerate(data):
            qtext = str(q.get("question", q.get("text", ""))).strip()
            opts = q.get("options") or list(q.get("distribution", {}).keys()) or ["Yes", "No"]
            out.append({"id": str(q.get("id", f"Q{i+1}")), "question": qtext, "options": list(opts)})
        return out

    if isinstance(data, dict):
        out = []
        for i, (qt, obj) in enumerate(data.items()):
            opts = []
            if isinstance(obj, dict):
                if "options" in obj:
                    o = obj["options"]
                    if isinstance(o, list):
                        opts = [str(x) if isinstance(x, str) else str(x.get("text", "")) for x in o]
                    elif isinstance(o, dict):
                        opts = [str(v) for v in o.values()]
                elif "distribution" in obj and isinstance(obj["distribution"], dict):
                    opts = [str(k) for k in obj["distribution"].keys()]
            if not opts:
                opts = ["Yes", "No"]
            out.append({"id": f"Q{i+1}", "question": str(qt).strip(), "options": opts})
        return out

    raise ValueError("Unsupported questions JSON format")

# Retriever (Whoosh BM25) 

class RetrieverAgent:
    """
    Sparse retriever using Whoosh BM25F over fields: 'title', 'text'.
    Expects the index to live at: <output_dir>/bm25_index (created by index.build).
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        set_seed(int(cfg.get("seed", 42)))

        idx_root = cfg["index"]["output_dir"]
        self.bm25_dir = os.path.join(idx_root, "bm25_index")  
        if not windex.exists_in(self.bm25_dir):
            raise FileNotFoundError(
                f"Whoosh index not found at {self.bm25_dir}. "
                "Run: python -m index.build --config config.yaml --api_key dummy"
            )
        self.ix = windex.open_dir(self.bm25_dir)

        rcfg = cfg.get("retrieval", {}) or {}
        self.sparse_topk = int(rcfg.get("sparse_topk", 400))

    def _search(self, query_text: str, k: int) -> List[Tuple[str, float]]:
        with self.ix.searcher(weighting=BM25F()) as searcher:
            parser = MultifieldParser(["title", "text"], schema=self.ix.schema, group=OrGroup)
            q = parser.parse(query_text.lower())
            res = searcher.search(q, limit=k)
            out = [(r["id"], float(r.score)) for r in res]
        # stable
        out.sort(key=lambda t: (-t[1], t[0]))
        return out[:k]

    def retrieve(self, query_text: str) -> List[Tuple[str, float]]:
        return self._search(query_text, self.sparse_topk)

#Aggregator 

class AggregatorAgent:
    """
    Aggregates per-option evidence into probabilities + exactly-K supports.

    For each option:
      - Query = "{question} {option}" (join configurable)
      - Score = sum of top-M doc scores, with lexical boosts:
          * token overlap bonus (token_weight)
          * phrase match bonus (phrase_bonus)
      - Apply temperature softmax; renormalize to 1±1e-6
      - Supports: round-robin across option lists; pad deterministically to K.
    """

    def __init__(self, cfg: Dict[str, Any], retriever: RetrieverAgent):
        self.cfg = cfg
        self.retriever = retriever
        rcfg = cfg.get("run", {}) or {}

        self.supports_k = int(rcfg.get("supports_k", 100))
        self.temperature = float(rcfg.get("temperature", 0.9))
        self.option_join = str(rcfg.get("option_query_join", " "))
        self.top_m_per_option = int(rcfg.get("top_m_per_option", 25))
        self.token_weight = float(rcfg.get("token_weight", 0.0))
        self.phrase_bonus = float(rcfg.get("phrase_bonus", 0.0))

    @staticmethod
    def _toks(s: str) -> List[str]:
        return [t for t in ''.join(ch.lower() if ch.isalnum() else ' ' for ch in s).split() if t]

    def score_options(self, question: str, options: List[str]) -> Tuple[np.ndarray, List[str]]:
        per_scores: List[float] = []
        per_lists: List[List[Tuple[str, float]]] = []

        for opt in options:
            q_text = f"{question}{self.option_join}{opt}".strip()
            cand = self.retriever.retrieve(q_text)  # [(doc_id, base_score), ...]

            # lexical boosting on retrieved docs
            boosted: List[Tuple[str, float]] = []
            if (self.token_weight > 0 or self.phrase_bonus > 0) and self.retriever.ix is not None:
                opt_tokens = set(self._toks(opt))
                phrase = opt.lower()
                with self.retriever.ix.searcher() as s:
                    for did, base in cand:
                        try:
                            row = s.document(id=did)  # stored fields: id, title, text
                            txt = ((row.get("title", "") or "") + " " + (row.get("text", "") or "")).lower()
                        except Exception:
                            txt = ""
                        overlap = sum(1 for t in opt_tokens if t and t in txt)
                        bonus = self.token_weight * overlap
                        if phrase and phrase in txt:
                            bonus += self.phrase_bonus
                        boosted.append((did, float(base + bonus)))
            else:
                boosted = cand

            boosted.sort(key=lambda x: (-x[1], x[0]))
            per_lists.append(boosted)

            top_items = boosted[: self.top_m_per_option]
            score = float(sum(s for _, s in top_items)) if top_items else 0.0
            per_scores.append(score)

        # probs
        raw = np.array(per_scores, dtype=np.float64)
        probs = normalize_probs(softmax_temperature(raw, self.temperature))

        # supports (round-robin across lists, dedup), pad to exactly K
        K = self.supports_k
        supports: List[str] = []
        seen = set()
        ptr = [0] * len(options)

        while len(supports) < K:
            progressed = False
            for i, lst in enumerate(per_lists):
                while ptr[i] < len(lst) and lst[ptr[i]][0] in seen:
                    ptr[i] += 1
                if ptr[i] < len(lst):
                    did = lst[ptr[i]][0]
                    if did not in seen:
                        supports.append(did); seen.add(did)
                    ptr[i] += 1
                    progressed = True
                    if len(supports) >= K:
                        break
            if not progressed:
                break

        # pad using a generic query if needed
        if len(supports) < K:
            global_pool = self.retriever.retrieve(question)
            for did, _ in global_pool:
                if did not in seen:
                    supports.append(did); seen.add(did)
                if len(supports) >= K:
                    break

        # as last resort, enumerate all stored ids lexicographically
        if len(supports) < K and self.retriever.ix is not None:
            with self.retriever.ix.searcher() as s:
                all_ids = sorted([str(d["id"]) for d in s.all_stored_fields()])
            for did in all_ids:
                if did not in seen:
                    supports.append(did); seen.add(did)
                if len(supports) >= K:
                    break

        supports = supports[:K]
        return probs, supports

# Main 

def main():
    import argparse
    ap = argparse.ArgumentParser(description="MAS Runner (BM25 + lexical boosting)")
    ap.add_argument("--config", required=True, help="Path to config.yaml")
    ap.add_argument("--api_key", required=False, help="Accepted but unused if offline")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    set_seed(int(cfg.get("seed", 42)))

    # paths
    qpath = cfg["data"]["questions_path"]
    out_csv = cfg["run"]["output_csv"]

    # init agents
    retriever = RetrieverAgent(cfg)
    aggregator = AggregatorAgent(cfg, retriever)

    # load questions
    questions = load_questions(qpath)

    rows = []
    for q in questions:
        qtext = q["question"]
        options = q["options"] or ["Yes", "No"]
        probs, supports = aggregator.score_options(qtext, options)

        # build distribution map
        dist_map = {options[i]: float(probs[i]) for i in range(len(options))}
        s = sum(dist_map.values())
        if abs(s - 1.0) > 1e-6:
            dist_map = {k: float(v / s) for k, v in dist_map.items()}

        rows.append((qtext, dist_map, supports))

    write_csv(out_csv, rows)
    print(f"✅ Wrote CSV to {out_csv} with {len(rows)} rows.")

if __name__ == "__main__":
    main()
