
"""
index/build.py — Hybrid (BM25 + optional FAISS) 

Config keys used:
  seed: 42
  data:
    documents_path: ./data/mini_documents.jsonl
    embeddings_npz: ./data/id_to_embedding.npz
  index:
    output_dir: ./artifacts/index
  retrieval:
    use_sparse: true
    use_dense: false   
"""

import os
import json
import yaml
import shutil
import random
import numpy as np
from typing import Dict, Any, List

from whoosh import index as windex
from whoosh.fields import Schema, ID, TEXT
from whoosh.analysis import StemmingAnalyzer
from whoosh.scoring import BM25F


# Utilities 
def set_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def fresh_dir(path: str):
    """Remove and recreate a directory (used to avoid stale Whoosh artifacts)."""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


# Document loading 
def load_documents(jsonl_path: str) -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = str(obj.get("id", ""))
            title = (obj.get("title", "") or "").strip()
            text = (obj.get("text", "") or "").strip()
            docs.append({"id": doc_id, "title": title, "text": text})
    return docs


# Dense helpers
def _load_embeddings(npz_path: str):
    """
    Supports two common layouts:
      (A) keys: 'ids' (array of strings) and 'embeddings' (N x d float array)
      (B) dict-like mapping: { id_str -> 1D float array }
    Returns: (ids: np.ndarray[str], embs: np.ndarray[float32])
    """
    data = np.load(npz_path, allow_pickle=True)
    # Case A
    if isinstance(data, np.lib.npyio.NpzFile) and "ids" in data.files and "embeddings" in data.files:
        ids = data["ids"]
        embs = data["embeddings"].astype("float32")
        return ids, embs
    # Case B (mapping)
    try:
        keys = list(data.keys())
        
        ids = np.array(keys)
        embs = np.vstack([data[k] for k in keys]).astype("float32")
        return ids, embs
    except Exception:
        raise ValueError(
            "Unsupported embedding NPZ format. Expect keys ('ids','embeddings') "
            "or mapping {id -> vector}."
        )


def _maybe_build_faiss(cfg: Dict[str, Any], out_dir: str):
    """Build FAISS index only if retrieval.use_dense is true."""
    rcfg = cfg.get("retrieval", {}) or {}
    use_dense = bool(rcfg.get("use_dense", False))
    if not use_dense:
        print("ℹ️ Dense retrieval disabled in config; skipping FAISS build.")
        return

    
    try:
        import faiss  
    except Exception as e:
        raise RuntimeError(
            "Dense index requested but faiss-cpu is not available. "
            "Install faiss-cpu or set retrieval.use_dense: false"
        ) from e

    embeddings_path = cfg["data"]["embeddings_npz"]
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Missing embeddings file: {embeddings_path}")

    ids, embs = _load_embeddings(embeddings_path)
    if embs.ndim != 2:
        raise ValueError("Embeddings must be a 2D array (N x d).")

    dim = embs.shape[1]
    
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = (embs / norms).astype("float32")

    dense_out = os.path.join(out_dir, "dense.index")
    ids_out = os.path.join(out_dir, "doc_ids.npy")

    index_flat = faiss.IndexFlatIP(dim)  
    index_flat.add(embs)
    faiss.write_index(index_flat, dense_out)
    np.save(ids_out, ids)

    print(f"✅ FAISS dense index built (dim={dim}, count={len(ids)})")


# Main builder 
def build_indexes(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg.get("seed", 42)))

    docs_path = cfg["data"]["documents_path"]
    out_root = cfg["index"]["output_dir"]
    ensure_dir(out_root)
    print(f"📂 Building indexes to: {out_root}")

    # Load corpus
    docs = load_documents(docs_path)
    print(f"✅ Loaded {len(docs)} documents from {docs_path}")

    # Build Whoosh BM25 
    bm25_dir = os.path.join(out_root, "bm25_index")
    
    fresh_dir(bm25_dir)

    schema = Schema(
        id=ID(stored=True, unique=True),
        title=TEXT(stored=True, analyzer=StemmingAnalyzer()),
        text=TEXT(stored=True, analyzer=StemmingAnalyzer()),
    )
    ix = windex.create_in(bm25_dir, schema)
    writer = ix.writer(limitmb=256)  

    for d in docs:
        writer.add_document(
            id=d["id"],
            title=d["title"].lower(),
            text=d["text"].lower(),
        )
    writer.commit()
    print(f"✅ BM25 index built with {len(docs)} docs")

    # Build FAISS 
    _maybe_build_faiss(cfg, out_root)

    print("🎯 Hybrid index build completed successfully.")


# Entry point 
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid Index Builder for MAS Assignment")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--api_key", required=False, help="Accepted for spec compatibility; unused here.")
    args = parser.parse_args()

    build_indexes(args.config)
