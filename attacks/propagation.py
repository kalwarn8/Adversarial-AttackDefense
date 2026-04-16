"""
Milestone 3: Constraint-Aware Adversarial Attack with Feature Propagation.

Feature dependency graph built from BOTH:
  - Pearson Correlation  → captures linear relationships (numerical features)
  - Mutual Information   → captures non-linear relationships (categorical features)

Combined score = 0.5 * correlation + 0.5 * mutual_information
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mutual_info_score
from attacks.constraints import build_constraint_mask, apply_constraints


# ── 1. Build the feature dependency graph ─────────────────────────────────────

def build_dependency_graph(X_train_np, feature_names, corr_threshold=0.10):
    """
    Builds adjacency matrix using Pearson Correlation + Mutual Information.

    Steps:
      1. Pearson Correlation  — linear relationships
      2. Mutual Information   — non-linear + categorical relationships
      3. Combined 50/50       — complete picture of dependencies

    Returns:
        adj       : np.ndarray (F, F) — normalized adjacency for propagation
        abs_corr  : np.ndarray (F, F) — raw correlation values
        mi_matrix : np.ndarray (F, F) — raw MI values
    """
    n_features = X_train_np.shape[1]

    # ── Step 1: Pearson Correlation ───────────────────────────────────────────
    print(f"[Graph] Step 1/3 — Computing Pearson correlation...")
    with np.errstate(invalid="ignore", divide="ignore"):
        corr_matrix = np.corrcoef(X_train_np.T)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    abs_corr = np.abs(corr_matrix).astype(np.float32)
    np.fill_diagonal(abs_corr, 0.0)

    # ── Step 2: Mutual Information ────────────────────────────────────────────
    # MI captures non-linear relationships that Pearson correlation misses.
    # Especially important for one-hot categorical features like
    # workclass, education, occupation — which are mutually exclusive.
    print(f"[Graph] Step 2/3 — Computing Mutual Information...")

    # Discretize features into bins for MI computation
    n_bins = 5
    X_binned = np.zeros_like(X_train_np, dtype=int)
    for i in range(n_features):
        col = X_train_np[:, i]
        bins = np.linspace(col.min(), col.max() + 1e-10, n_bins + 1)
        X_binned[:, i] = np.digitize(col, bins) - 1

    mi_matrix = np.zeros((n_features, n_features), dtype=np.float32)

    # Only compute MI for pairs that already have weak correlation
    # This avoids computing all 104x104 = 10816 pairs (too slow)
    candidate_pairs = np.argwhere(abs_corr >= corr_threshold * 0.5)
    for i, j in candidate_pairs:
        if i >= j:
            continue
        mi = mutual_info_score(X_binned[:, i], X_binned[:, j])
        mi_matrix[i, j] = mi
        mi_matrix[j, i] = mi

    # Normalize MI to [0, 1]
    mi_max = mi_matrix.max()
    if mi_max > 0:
        mi_matrix = mi_matrix / mi_max

    # ── Step 3: Combine correlation + MI 50/50 ───────────────────────────────
    print(f"[Graph] Step 3/3 — Combining correlation + MI...")
    combined = 0.5 * abs_corr + 0.5 * mi_matrix
    np.fill_diagonal(combined, 0.0)

    # Threshold and normalize
    adj = np.where(combined >= corr_threshold, combined, 0.0).astype(np.float32)
    np.fill_diagonal(adj, 0.0)

    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    adj = adj / row_sums

    n_edges = (adj > 0).sum()
    print(f"[Graph] Done!")
    print(f"         Nodes     : {n_features} features")
    print(f"         Edges     : {n_edges} connections")
    print(f"         Method    : 50% Pearson correlation + 50% Mutual Information")
    print(f"         Threshold : {corr_threshold}\n")

    return adj, abs_corr, mi_matrix


# ── 2. Propagate perturbation ─────────────────────────────────────────────────

def propagate_perturbation(delta, adj, propagation_strength=0.5):
    """
    Spreads perturbation delta through the dependency graph.

    When feature i is perturbed, neighbor j receives:
        delta_j += propagation_strength * adj[i,j] * delta_i
    """
    adj_tensor = torch.tensor(adj, dtype=torch.float32)
    propagated = propagation_strength * torch.matmul(delta, adj_tensor)
    return delta + propagated


# ── 3. Propagated FGSM ───────────────────────────────────────────────────────

def fgsm_propagated(model, X, y, epsilon=0.5,
                    adj=None, feature_names=None,
                    propagation_strength=0.5):
    """
    FGSM + constraint mask + dependency propagation.

    Steps:
      1. Standard FGSM gradient step
      2. Apply constraint mask — freeze immutable features (sex, race)
      3. Propagate through dependency graph (corr + MI)
      4. Apply domain constraints (clip age, hours etc.)
    """
    X_in = X.clone().detach().requires_grad_(True)

    model.eval()
    outputs = model(X_in)
    loss = F.cross_entropy(outputs, y)
    model.zero_grad()
    loss.backward()

    grad = X_in.grad.data
    delta = epsilon * grad.sign()

    # Freeze immutable features
    if feature_names is not None:
        mask = build_constraint_mask(feature_names)
        delta = delta * mask

    # Propagate through graph
    if adj is not None:
        delta = propagate_perturbation(delta, adj, propagation_strength)

    perturbed = X + delta

    # Enforce domain constraints
    if feature_names is not None:
        perturbed = apply_constraints(perturbed, X, feature_names)

    return perturbed.detach()


# ── 4. Propagated PGD ────────────────────────────────────────────────────────

def pgd_propagated(model, X, y, epsilon=0.4, alpha=0.04, iters=20,
                   adj=None, feature_names=None,
                   propagation_strength=0.3):
    """
    PGD + constraint mask + dependency propagation.

    Design: run full PGD first to find strongest direction,
    then propagate the final accumulated delta ONCE through graph.
    Propagating inside each iteration dilutes the attack.
    """
    perturbed = X.clone().detach()
    model.eval()

    # Step 1: Run full PGD
    for _ in range(iters):
        perturbed.requires_grad_(True)

        outputs = model(perturbed)
        loss = F.cross_entropy(outputs, y)
        model.zero_grad()
        loss.backward()

        grad = perturbed.grad.data
        step = alpha * grad.sign()

        # Freeze immutable features at every step
        if feature_names is not None:
            mask = build_constraint_mask(feature_names)
            step = step * mask

        perturbed = perturbed.detach() + step
        eta = torch.clamp(perturbed - X, min=-epsilon, max=epsilon)
        perturbed = (X + eta).detach()

    # Step 2: Propagate final delta ONCE
    total_delta = perturbed - X
    if adj is not None:
        total_delta = propagate_perturbation(total_delta, adj, propagation_strength)

    perturbed = X + total_delta

    # Step 3: Apply constraints once at the end
    if feature_names is not None:
        perturbed = apply_constraints(perturbed, X, feature_names)

    return perturbed.detach()
