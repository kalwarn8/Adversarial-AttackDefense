"""
attacks/propagation.py
-----------------------
Milestone 3: Constraint-Aware Adversarial Attack with Feature Propagation.

The idea (from the proposal):
  When an attacker changes one feature, related features must shift too
  to keep the record realistic. We model this using a feature dependency
  graph built from correlation + mutual information on the training data.

  Perturbation on feature i  →  propagates to neighbor j
  with weight proportional to the edge strength (correlation/MI).

Professor feedback addressed:
  - Constraints are enforced via constraints.py (immutable/bounded/direction)
  - Propagation respects those same constraints after spreading
"""

import numpy as np
import torch
import torch.nn.functional as F
from attacks.constraints import build_constraint_mask, apply_constraints


# ── 1. Build the feature dependency graph ─────────────────────────────────────

def build_dependency_graph(X_train_np, feature_names, corr_threshold=0.10):
    """
    Builds a (n_features x n_features) adjacency matrix from training data.

    For each pair (i, j):
      - Computes absolute Pearson correlation
      - Edge weight = correlation if >= corr_threshold, else 0

    Args:
        X_train_np      : np.ndarray, shape (N, F) — training data (scaled)
        feature_names   : list of str, length F
        corr_threshold  : minimum correlation to include an edge

    Returns:
        adj : np.ndarray shape (F, F), values in [0, 1]
              adj[i, j] = how strongly feature j should shift when i is perturbed
    """
    n_features = X_train_np.shape[1]
    adj = np.zeros((n_features, n_features), dtype=np.float32)

    # Compute correlation matrix (handles NaN gracefully)
    # np.corrcoef needs features as rows
    with np.errstate(invalid="ignore", divide="ignore"):
        corr_matrix = np.corrcoef(X_train_np.T)  # shape (F, F)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    abs_corr = np.abs(corr_matrix)

    # Zero out weak edges and self-loops
    adj = np.where(abs_corr >= corr_threshold, abs_corr, 0.0)
    np.fill_diagonal(adj, 0.0)  # no self-propagation

    # Row-normalize so each row sums to ≤ 1
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)  # avoid div by zero
    adj = adj / row_sums

    print(f"[Graph] Built dependency graph: {n_features} nodes, "
          f"{(adj > 0).sum()} edges (threshold={corr_threshold})")

    return adj


def propagate_perturbation(delta, adj, propagation_strength=0.5):
    """
    Spreads a perturbation delta through the dependency graph.

    For each feature i that was perturbed, its neighbors j receive:
        delta_j += propagation_strength * adj[i, j] * delta_i

    Args:
        delta                 : torch.Tensor shape (N, F)
        adj                   : np.ndarray shape (F, F)
        propagation_strength  : how much of the perturbation spreads (0–1)

    Returns:
        torch.Tensor shape (N, F) — propagated perturbation
    """
    adj_tensor = torch.tensor(adj, dtype=torch.float32)
    # delta: (N, F) @ adj.T: (F, F) → (N, F)
    propagated = propagation_strength * torch.matmul(delta, adj_tensor)
    return delta + propagated


# ── 2. Constraint-Aware FGSM with Propagation ─────────────────────────────────

def fgsm_propagated(model, X, y, epsilon=0.3,
                    adj=None, feature_names=None,
                    propagation_strength=0.5):
    """
    FGSM attack + feature dependency propagation + constraint enforcement.

    Steps:
      1. Standard FGSM gradient step
      2. Propagate perturbation through dependency graph
      3. Enforce domain constraints (immutable, bounded, direction)

    Args:
        model                : trained PyTorch model
        X                    : torch.Tensor (N, F), clean input
        y                    : torch.Tensor (N,), true labels
        epsilon              : perturbation magnitude
        adj                  : dependency graph (F, F) np.ndarray
        feature_names        : list of feature names for constraint enforcement
        propagation_strength : how strongly perturbation spreads

    Returns:
        perturbed X as torch.Tensor (N, F)
    """
    X_in = X.clone().detach().requires_grad_(True)

    outputs = model(X_in)
    loss = F.cross_entropy(outputs, y)
    model.zero_grad()
    loss.backward()

    grad = X_in.grad.data
    delta = epsilon * grad.sign()

    # Apply constraint mask — zero out immutable features
    if feature_names is not None:
        mask = build_constraint_mask(feature_names)
        delta = delta * mask

    # Propagate through dependency graph
    if adj is not None:
        delta = propagate_perturbation(delta, adj, propagation_strength)

    perturbed = X + delta

    # Enforce domain constraints
    if feature_names is not None:
        perturbed = apply_constraints(perturbed, X, feature_names)

    return perturbed.detach()


# ── 3. Constraint-Aware PGD with Propagation ──────────────────────────────────

def pgd_propagated(model, X, y, epsilon=0.2, alpha=0.02, iters=10,
                   adj=None, feature_names=None,
                   propagation_strength=0.5):
    """
    PGD attack + feature dependency propagation + constraint enforcement.

    At each iteration:
      1. Gradient step (alpha)
      2. Propagate perturbation
      3. Project back into epsilon-ball
      4. Enforce domain constraints

    Args:
        model                : trained PyTorch model
        X                    : torch.Tensor (N, F)
        y                    : torch.Tensor (N,)
        epsilon              : max total perturbation
        alpha                : step size per iteration
        iters                : number of PGD steps
        adj                  : dependency graph np.ndarray (F, F)
        feature_names        : list of feature names
        propagation_strength : spread factor

    Returns:
        perturbed X as torch.Tensor (N, F)
    """
def pgd_propagated(model, X, y, epsilon=0.2, alpha=0.02, iters=10,
                   adj=None, feature_names=None,
                   propagation_strength=0.5):
    """
    PGD attack + feature dependency propagation + constraint enforcement.

    Key design decision:
      - Run standard PGD iterations to find the strongest adversarial direction
      - Apply propagation ONCE at the end to the total accumulated delta
      - Apply constraints ONCE at the end
      
    Propagating inside every iteration dilutes the attack because
    spreading perturbations across many features reduces the gradient
    signal at each step. Propagating the final delta once preserves
    attack strength while still making perturbations realistic.
    """
    perturbed = X.clone().detach()

    # Step 1: Run standard PGD to find strongest perturbation direction
    for _ in range(iters):
        perturbed.requires_grad_(True)

        outputs = model(perturbed)
        loss = F.cross_entropy(outputs, y)
        model.zero_grad()
        loss.backward()

        grad = perturbed.grad.data
        step = alpha * grad.sign()

        # Zero out immutable features at every step
        if feature_names is not None:
            mask = build_constraint_mask(feature_names)
            step = step * mask

        perturbed = perturbed.detach() + step

        # Project back into epsilon-ball
        eta = torch.clamp(perturbed - X, min=-epsilon, max=epsilon)
        perturbed = (X + eta).detach()

    # Step 2: Get the total accumulated delta from PGD
    total_delta = perturbed - X

    # Step 3: Propagate the FINAL delta through dependency graph (once)
    if adj is not None:
        total_delta = propagate_perturbation(total_delta, adj, propagation_strength)

    perturbed = X + total_delta

    # Step 4: Apply domain constraints once at the very end
    if feature_names is not None:
        perturbed = apply_constraints(perturbed, X, feature_names)

    return perturbed.detach()
