# Multi-Agent Retrieval & Evidence Pipeline

A Python-based multi-agent information retrieval system designed to retrieve, rank and evaluate supporting evidence for complex survey-style questions.

The system combines lexical search, semantic re-ranking and confidence calibration to improve evidence quality and produce more reliable outputs.

## Key Highlights

- Built a modular multi-agent pipeline with dedicated retrieval, query expansion, aggregation and calibration components.
- Combined BM25 lexical retrieval with dense semantic re-ranking to improve evidence relevance.
- Improved Mean Average Precision (MAP) from **0.735 to 0.867 (+17.9%)** during evaluation.
- Reduced mean entropy from **0.416 to 0.220**, improving confidence and output consistency.
- Implemented structured configuration and evaluation workflows using Python and YAML.
- Evaluated retrieval performance using MAP, Recall, nDCG and Jensen-Shannon Divergence.

## Architecture

The system is organised into four specialised components:

- **Retriever Agent** — retrieves candidate documents using BM25 lexical search.
- **Expansion Agent** — expands queries and identifies relevant stance information.
- **Aggregator Agent** — merges, deduplicates and ranks supporting evidence.
- **Calibrator Agent** — evaluates uncertainty and confidence in generated outputs.

This modular design allows individual retrieval and reasoning components to be evaluated and improved independently.

## Retrieval Pipeline

1. Build a BM25 index from the document collection.
2. Retrieve candidate documents for each query.
3. Expand ambiguous queries where required.
4. Apply dense semantic re-ranking to prioritise relevant evidence.
5. Aggregate and remove redundant results.
6. Calibrate confidence scores and evaluate final outputs.

## Evaluation Results

| Metric | Baseline | Enhanced |
|---|---:|---:|
| MAP | 0.735 | 0.867 |
| Mean Entropy | 0.416 | 0.220 |
| Mean Max Probability | 0.735 | 0.867 |
| JS Score | 0.257 | 0.207 |

The enhanced retrieval and calibration approach improved evidence precision and consistency while maintaining strong retrieval coverage.

## Tech Stack

- **Language:** Python
- **Configuration:** YAML
- **Retrieval:** BM25, Dense Re-ranking, Semantic Retrieval
- **AI / NLP:** Multi-Agent Systems, Query Expansion, Embeddings
- **Evaluation:** MAP, Recall, nDCG, Jensen-Shannon Divergence, Entropy
- **Development:** Git, GitHub

## Project Structure

```text
multi-agent-retrieval-evidence-pipeline/
├── index/
│   └── build.py
├── mas_survey/
│   └── run.py
├── probes/
│   ├── eval_calib.py
│   ├── eval_ir.py
│   ├── dump_ranked_ids.py
│   └── dump_topk_for_annotation.py
├── config.yaml
├── config_peaky.yaml
├── evaluate_js.py
├── evaluate_map.py
├── requirements.txt
└── README.md