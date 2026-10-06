# Multi-Agent Retrieval & Evidence Pipeline

A Python-based multi-agent information retrieval system designed to retrieve, rank and evaluate supporting evidence for complex survey-style questions.

The pipeline combines BM25 lexical retrieval, dense semantic re-ranking and confidence calibration to improve evidence quality and retrieval reliability.

## Key Result

**Improved Mean Average Precision (MAP) from 0.735 to 0.867 (+17.9%) while reducing mean entropy from 0.416 to 0.220.**

## Key Highlights

- Built a modular multi-agent pipeline for retrieval, query expansion, evidence aggregation and confidence calibration.
- Combined **BM25 lexical retrieval with dense semantic re-ranking** to improve evidence relevance.
- Improved **MAP from 0.735 to 0.867 (+17.9%)** during evaluation.
- Reduced **mean entropy from 0.416 to 0.220**, improving confidence consistency.
- Implemented structured configuration and evaluation workflows using **Python and YAML**.
- Evaluated retrieval performance using **MAP, Recall, nDCG and Jensen-Shannon Divergence**.

## Architecture

The system is organised into four specialised components:

- **Retriever Agent** — retrieves candidate documents using BM25 lexical search.
- **Expansion Agent** — expands queries and identifies relevant stance information.
- **Aggregator Agent** — merges, deduplicates and ranks supporting evidence.
- **Calibrator Agent** — evaluates uncertainty and confidence in the final outputs.

The modular architecture allows individual retrieval and reasoning components to be evaluated, modified and improved independently.

## Retrieval Pipeline

1. Build a BM25 search index from the document collection.
2. Retrieve an initial set of candidate documents for each query.
3. Expand ambiguous queries where required.
4. Apply dense semantic re-ranking to prioritise relevant evidence.
5. Aggregate and remove redundant or off-topic results.
6. Calibrate confidence scores and evaluate the final outputs.

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
- **Information Retrieval:** BM25, Dense Re-ranking, Semantic Retrieval
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
```

## Running the Project

Create a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Build the retrieval index:

```bash
python -m index.build --config config_peaky.yaml --api_key dummy_api
```

Run the multi-agent pipeline:

```bash
python -m mas_survey.run --config config_peaky.yaml --api_key dummy_api
```

## Repository Notes

The full competition dataset and generated retrieval artifacts are intentionally excluded from this public portfolio repository due to file-size and redistribution considerations.

This project was developed as part of postgraduate coursework at RMIT University and has been reorganised as a portfolio project to highlight the system architecture, implementation and evaluation methodology.

## What I Learned

This project strengthened my practical experience in information retrieval, multi-agent system design, retrieval evaluation and iterative model improvement.

It also involved balancing retrieval accuracy, evidence quality, confidence and computational constraints rather than optimising a single metric in isolation.

## Author

**Ragulkarthik Sundar**  
Master of Data Science — RMIT University  
Melbourne, Australia