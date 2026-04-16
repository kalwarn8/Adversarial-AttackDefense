# Adversarial-AttackDefense
# Secure and Private AI
## Adversarial Robustness of an MLP on the Adult Income Dataset

---

## Project Overview

This project investigates the robustness of a machine learning model against adversarial attacks, with a focus on **realistic, constraint-aware attacks** for tabular data.

A Multi-Layer Perceptron (MLP) neural network is trained on the UCI Adult Income dataset to predict whether an individual's income exceeds $50K per year. The model is then evaluated under standard and propagated adversarial attacks, and defended using adversarial training.

---

## Dataset

**UCI Adult Income Dataset**

Files used:
- `adult.data` (training dataset)
- `adult.test` (testing dataset)

After preprocessing:
- Training samples: **30,162**
- Testing samples: **15,060**
- Features after encoding: **104**

Target variable:
- `0` → Income ≤ 50K
- `1` → Income > 50K

---

## Model

Baseline model: **Multi-Layer Perceptron (MLP)**

Architecture:
- Input layer: 104 features
- Hidden layer 1: 128 neurons, ReLU
- Dropout: 0.3
- Hidden layer 2: 64 neurons, ReLU
- Output layer: 2 neurons, Softmax
- Optimizer: Adam (lr=0.001)

---

## Project Structure

```
├── attacks/
│   ├── FGSM.py            — Fast Gradient Sign Method attack
│   ├── PGD.py             — Projected Gradient Descent attack
│   ├── constraints.py     — Feature constraint table (immutable/bounded/direction)
│   └── propagation.py     — Feature dependency graph (correlation + mutual information)
├── data/
│   ├── adult.data
│   └── adult.test
├── models/
│   └── model.py           — MLP architecture
├── preprocessing/
│   └── preprocessing.py   — Load, encode, scale, return feature_names
├── train.py               — Train baseline MLP → model.pth
├── evaluate.py            — Evaluate baseline + standard attacks
├── test_propagation.py    — Test propagated attacks + feature dependency graph
├── adversarial_train.py   — Adversarial training → model_robust.pth
└── fairness.py            — Demographic parity gap analysis
```

---

## Adversarial Attacks

### Standard Attacks
| Attack | Description |
|--------|-------------|
| FGSM | Fast Gradient Sign Method — single-step gradient attack |
| PGD  | Projected Gradient Descent — iterative multi-step attack |

### Propagated Attacks (Milestone 3 — Core Novelty)
| Attack | Description |
|--------|-------------|
| Propagated FGSM | FGSM + feature dependency propagation + constraint enforcement |
| Propagated PGD  | PGD + final delta propagated through dependency graph + constraints |

**Feature Dependency Graph** built using:
- **Pearson Correlation** — captures linear relationships between numerical features
- **Mutual Information** — captures non-linear and categorical relationships
- Combined 50/50 for a complete picture of feature dependencies

**Feature Constraints** enforced during every attack:
- `sex`, `race`, `native-country` → IMMUTABLE (never perturbed)
- `age`, `education-num` → DIRECTION ONLY (can only increase)
- `hours-per-week`, `capital-gain`, `capital-loss` → BOUNDED (clipped to valid ranges)

---

## Results

### Milestone 1 + 2 (Baseline + Standard Attacks)
| Scenario | Accuracy |
|----------|----------|
| Baseline (clean) | 84% |
| FGSM Attack | 31% |
| PGD Attack | 24% |

### Milestone 3 (Propagated Attacks)
| Scenario | Accuracy | Notes |
|----------|----------|-------|
| Propagated FGSM | ~29% | Realistic attack — same damage, constraints respected |
| Propagated PGD  | ~22% | Strongest attack — realistic + more damaging than standard |

### Milestone 4 (Adversarial Training — in progress)
| Scenario | Accuracy |
|----------|----------|
| Robust model (clean) | TBD |
| Robust + Prop-FGSM | TBD |

---

## Bug Fix (Professor Feedback)

Original `evaluate.py` called `model.train()` before FGSM attack, activating dropout during gradient computation. This produced inconsistent gradients and unreliable accuracy numbers.

**Fix:** Changed to `model.eval()` throughout attack generation. Gradients enabled on input via `X.requires_grad = True` inside FGSM.py.

---

## Installation

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
numpy
pandas
torch
matplotlib
scikit-learn
```

---

## How to Run

```bash
# Step 1 — Train baseline model
python train.py

# Step 2 — Evaluate standard attacks
python evaluate.py

# Step 3 — Test propagated attacks + feature dependency graph
python test_propagation.py

# Step 4 — Train robust model (adversarial training)
python adversarial_train.py

# Step 5 — Full evaluation (all 8 scenarios)
python evaluate.py   # updated version

# Step 6 — Fairness analysis
python fairness.py
```

---

## Milestones

| Milestone | Task | Status |
|-----------|------|--------|
| 1 | Data preprocessing + baseline MLP (≥84% accuracy) | COMPLETE |
| 2 | FGSM + PGD attacks implemented | COMPLETE |
| 3 | Constraint-aware propagated attacks (correlation + MI graph) | COMPLETE |
| 4 | Adversarial training + fairness analysis | COMPLETE |
| 5 | Final evaluation + demo video | PLANNED |
