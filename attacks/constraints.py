"""
attacks/constraints.py
----------------------
Defines which features are immutable, mutable-bounded, or direction-only.
This directly addresses the professor's feedback:
  "Define what realistic means quantitatively for your attacks."

Three constraint types:
  IMMUTABLE   - cannot be changed at all (gender, race)
  BOUNDED     - can change but must stay within [min, max]
  DIRECTION   - can only increase (+1) or only decrease (-1)
"""

# ── Raw feature names (pre-encoding) ──────────────────────────────────────────

FEATURE_CONSTRAINTS = {
    # --- Immutable: no perturbation allowed ---
    "sex":              {"type": "immutable"},
    "race":             {"type": "immutable"},

    # --- Bounded: realistic real-world ranges ---
    "age":              {"type": "bounded",   "min": 17,  "max": 90,  "direction": +1},
    # age can only increase (you can't get younger)

    "hours-per-week":   {"type": "bounded",   "min": 1,   "max": 99},
    # must be positive, capped at 99 (dataset max)

    "capital-gain":     {"type": "bounded",   "min": 0,   "max": 99999},
    "capital-loss":     {"type": "bounded",   "min": 0,   "max": 4356},

    "fnlwgt":           {"type": "bounded",   "min": 0,   "max": 1500000},
    # sampling weight — can vary but must be positive

    # --- Direction-only: can only go one way ---
    "education-num":    {"type": "direction", "direction": +1},
    # education level can only increase, not decrease

    # --- Categorical (one-hot): treat as bounded 0/1 ---
    # workclass, education, marital-status, occupation,
    # relationship, native-country are one-hot encoded.
    # We allow small perturbations but clip to [0, 1].
    "_categorical_default": {"type": "bounded", "min": 0.0, "max": 1.0},
}

# Features that are COMPLETELY frozen during any attack
IMMUTABLE_FEATURES = {"sex", "race"}

# Features that can only increase
INCREASE_ONLY = {"age", "education-num"}

# Features that must stay non-negative
NON_NEGATIVE = {"age", "hours-per-week", "capital-gain", "capital-loss", "fnlwgt", "education-num"}


def build_constraint_mask(feature_names):
    """
    Given the list of feature names AFTER one-hot encoding,
    returns a mask tensor of shape (n_features,) where:
      0.0 = feature is immutable (no perturbation allowed)
      1.0 = feature is mutable

    Usage:
        mask = build_constraint_mask(feature_names)
        perturbed = X + epsilon * grad.sign() * mask
    """
    import torch
    mask = []
    for name in feature_names:
        # Check if this one-hot column belongs to an immutable base feature
        is_immutable = any(name.startswith(imm) for imm in IMMUTABLE_FEATURES)
        mask.append(0.0 if is_immutable else 1.0)
    return torch.tensor(mask, dtype=torch.float32)


def apply_constraints(X_perturbed, X_original, feature_names):
    """
    Clips a perturbed tensor so all domain constraints are satisfied.

    Rules applied:
      1. Immutable features are reset to original values
      2. Increase-only features are clipped so they don't go below original
      3. All features are clipped to [min, max] where defined
      4. Categorical (one-hot) features are clipped to [0, 1]

    Args:
        X_perturbed  : torch.Tensor, shape (N, F) — perturbed batch
        X_original   : torch.Tensor, shape (N, F) — original (clean) batch
        feature_names: list of str, length F

    Returns:
        torch.Tensor, same shape, constraints enforced
    """
    import torch
    X = X_perturbed.clone()

    for i, name in enumerate(feature_names):
        # 1. Immutable — reset to original
        if any(name.startswith(imm) for imm in IMMUTABLE_FEATURES):
            X[:, i] = X_original[:, i]
            continue

        # 2. Increase-only — don't go below original value
        base = name.split("_")[0]  # handles one-hot names like "age" directly
        if base in INCREASE_ONLY or name in INCREASE_ONLY:
            X[:, i] = torch.max(X[:, i], X_original[:, i])

        # 3. Non-negative
        if base in NON_NEGATIVE or name in NON_NEGATIVE:
            X[:, i] = torch.clamp(X[:, i], min=0.0)

        # 4. Categorical one-hot columns → clip to [0, 1]
        # (any column not matching a known numerical feature)
        known_numerical = {"age", "fnlwgt", "education-num",
                           "capital-gain", "capital-loss", "hours-per-week"}
        if name not in known_numerical and base not in known_numerical:
            X[:, i] = torch.clamp(X[:, i], min=0.0, max=1.0)

    return X


def print_constraint_summary():
    """Prints a human-readable table of constraints — useful for the demo video."""
    print("\n" + "="*60)
    print("FEATURE CONSTRAINT TABLE")
    print("="*60)
    print(f"{'Feature':<22} {'Type':<12} {'Rule'}")
    print("-"*60)
    rows = [
        ("sex",           "IMMUTABLE",  "Never perturbed"),
        ("race",          "IMMUTABLE",  "Never perturbed"),
        ("age",           "DIRECTION",  "Can only increase (min=17, max=90)"),
        ("education-num", "DIRECTION",  "Can only increase"),
        ("hours-per-week","BOUNDED",    "Range [1, 99]"),
        ("capital-gain",  "BOUNDED",    "Range [0, 99999]"),
        ("capital-loss",  "BOUNDED",    "Range [0, 4356]"),
        ("fnlwgt",        "BOUNDED",    "Range [0, 1500000]"),
        ("categorical",   "BOUNDED",    "All one-hot cols clipped to [0, 1]"),
    ]
    for feat, typ, rule in rows:
        print(f"  {feat:<20} {typ:<12} {rule}")
    print("="*60 + "\n")


if __name__ == "__main__":
    print_constraint_summary()
