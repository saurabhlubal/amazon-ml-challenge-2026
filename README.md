# Amazon ML Challenge 2026: Business Entity Resolution

## Overview
This repository contains the complete, production-ready solution for the **Amazon ML Challenge 2026: Business Entity Resolution** task.
The objective is to accurately and efficiently match Source-1 business records against millions of candidate entities in Source-2 and Source-3 while strictly maintaining high precision and recall under bounded computational resources.

---

## 1. System Architecture

The pipeline implements an end-to-end multi-stage architecture:

```
Raw Multi-Source Data
       ↓
Multi-Representation Normalization (NFKC, Accents, Domain Stripping, Legal Suffixes, Address Parsing)
       ↓
Multi-Key Inverted Index Blocking (11 Deterministic Keys, In-Memory array('I') Index)
       ↓
Cheap Deterministic Candidate Prefilter (Country Filtering + Token/Signature Signal + Cap 80)
       ↓
Vectorized Feature Extraction (13 Pairwise Numeric Similarity Signals in NumPy / SIMD BLAS)
       ↓
Calibrated Lightweight Supervised Classifier (FastLogisticRegression, L2-regularized)
       ↓
Threshold Decision (Optimal F0.5 Threshold = 0.50)
       ↓
Streaming Candidate-Subset Enforcement & Submission Formatting
```

---

## 2. Benchmark & Decision Matrix (Phase 4)

We rigorously evaluated four configurations on identical representative validation data (5,000 Source-1 queries evaluated against a 167,362 candidate pool with 17,362 true matches):

| Configuration | Candidate Recall | Overall Match Recall | Macro F0.5 | Precision | Avg Candidates/S1 | P95 Candidates | Throughput (S1/s) | Total Runtime | Peak RAM | Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **A. Current Validated Pipeline** | 93.21% | 86.03% | 0.9267 | 0.9784 | 346.7 | 836 | 101.5 | 49.28s | 861 MB | Baseline |
| **B. DuckDB/Polars Only** | 93.21% | 86.03% | 0.9267 | 0.9784 | 346.7 | 836 | 32.8 | 152.23s | 1,983 MB | Rejected (3x slower due to DataFrame conversion overhead on huge candidate sets) |
| **C. Cheap Prefilter Only** | **92.40%** | **85.89%** | **0.9260** | **0.9791** | **67.6** | **80** | **440.4** | **11.35s** | **861 MB** | **SELECTED WINNER (4.34x speedup, 5.1x candidate reduction, higher precision)** |
| **D. Prefilter + DuckDB/Polars** | 92.40% | 85.89% | 0.9260 | 0.9791 | 67.6 | 80 | 191.0 | 26.18s | 1,578 MB | Viable, but 2.3x slower than pure NumPy/C vectorization |

### Key Decision Rationale:
- **Configuration C** reduced candidate volume by **5.1x** (from 346.7 to 67.6 per S1) while preserving **99.1% of candidate recall** and **99.8% of ground truth match recall**.
- Precision increased from **0.9784 to 0.9791**.
- Throughput increased by **4.34x** (from 101.5 to 440.4 S1/sec).
- Memory stayed strictly bounded (< 2.8 GB peak across entire test set).

---

## 3. Repository Structure

```
amazon-ml-challenge-2026/
├── business_entity_resolution/
│   └── src/
│       ├── normalization.py         # Unicode, phonetic, domain, address & shingle normalization
│       ├── blocking.py              # 11-key candidate blocking indexes
│       ├── prefilter.py             # Cheap deterministic candidate prefiltering & ranking
│       ├── features.py              # 13 pairwise feature extractors
│       ├── model.py                 # FastLogisticRegression model implementation
│       ├── evaluation.py            # Challenge-compliant Macro F0.5 evaluation
│       └── trained_model.json       # Production trained model weights, bias & threshold
├── sagemaker/
│   ├── vectorized_matcher.py        # CompactInvertedIndex & vectorized candidate scoring
│   ├── duckdb_matcher.py            # Polars/DuckDB vectorized matching implementation
│   ├── entrypoint.py                # SageMaker container entrypoint (bounded memory fallback)
│   └── launch_production.py         # SageMaker cluster job launcher
├── scripts/
│   ├── benchmark_prefilter.py       # Benchmark suite for prefilter configurations
│   ├── benchmark_four_configurations.py # Phase 4 benchmark matrix
│   ├── run_larger_validation.py     # 25,000 S1 validation test
│   ├── run_production_pipeline.py   # Full test production execution script
│   └── pipeline_utils.py            # Streaming IO, TSV formatting & ID utilities
├── student_resource/
│   └── utils/
│       └── validate_submission.py   # Official challenge submission validator
├── output/
│   ├── candidate_pairs.tsv          # Full test candidate pairs (1,732,544 rows)
│   └── matching_results.tsv         # Full test final match predictions (1,732,544 rows)
├── DOCUMENTATION.md                 # Complete challenge documentation template
├── requirements.txt                 # Exact environment dependencies
└── README.md
```

---

## 4. Reproducibility Instructions

### Prerequisites
- Python 3.10+ (tested on Python 3.14)
- 16 GB RAM

### Environment Setup
```bash
# Clone the repository and switch to the optimization branch
git clone https://github.com/saurabhlubal/amazon-ml-challenge-2026.git
cd amazon-ml-challenge-2026
git checkout feature/duckdb-optimization

# Initialize virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Reproducing Benchmark Matrix (Phase 4)
```bash
python scripts/benchmark_four_configurations.py
```

### Reproducing Full Test Submission
```bash
python scripts/run_production_pipeline.py \
    --test-dir student_resource/dataset/test \
    --output-dir output \
    --max-candidates 80
```

### Running Official Validation Check
```bash
python student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```
