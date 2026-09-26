# Amazon ML Challenge 2026 — Experiment Log

| Exp ID | Date/Time | Branch | Model | Features | Threshold | Local Macro F0.5 | Precision/Recall | Sampling Strategy | Status |
|---|---|---|---|---|---|---|---|---|---|
| E01 | 2026-09-26 19:37:14 | `feature/ml-matching` | LightGBM | 40 signals | Thresh=0.66 | **0.9962** | Prec=0.9984, Rec=0.9925 | Pos + Hard Negatives + Singleton Distractors | Keep |
| E02 | 2026-09-26 19:38:23 | `feature/ml-matching` | LightGBM | 40 signals | Thresh=0.46 | **0.9953** | Prec=0.9966, Rec=0.9927 | Pos + Hard Negatives + Singleton Distractors | Keep |
| E03 | 2026-09-26 20:01:23 | `feature/ml-matching` | LightGBM | 46 signals (cached) | Thresh=0.62 | **0.9974** | Prec=0.9984, Rec=0.9954 | Pos + Hard Negatives + Singleton Distractors | Keep |
