"""
What this script does step by step:
  1. Loads the UCI Adult dataset
  2. Builds the feature dependency graph
  3. Shows you the TOP 10 strongest feature relationships
  4. Runs propagated FGSM attack
  5. Compares accuracy: standard FGSM vs propagated FGSM
  6. Shows a sample of what actually changed in the features

"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from preprocessing.preprocessing import load_data
from models.model import MLP
from attacks.FGSM import fgsm_attack
from attacks.PGD import pgd_attack
from attacks.propagation import build_dependency_graph, fgsm_propagated, pgd_propagated
from attacks.constraints import print_constraint_summary, build_constraint_mask


def main():

    print("\n" + "="*60)
    print("STEP 1 — Load Data")
    print("="*60)
    X_train, X_test, y_train, y_test, feature_names = load_data("data/adult.data")

    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
    y_test_t  = torch.tensor(y_test.values, dtype=torch.long)

    print(f"\n  Features after encoding : {len(feature_names)}")
    print(f"  First 5 features        : {feature_names[:5]}")

    # ── Step 2: Show constraint summary ──────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 2 — Feature Constraints")
    print("="*60)
    print_constraint_summary()

    # Show which features are frozen (mask = 0)
    mask = build_constraint_mask(feature_names)
    frozen = [feature_names[i] for i in range(len(feature_names)) if mask[i] == 0]
    print(f"  Frozen features ({len(frozen)} total): {frozen[:8]}...")

    # ── Step 3: Build dependency graph ───────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 3 — Building Feature Dependency Graph")
    print("="*60)
    print("  Computing Pearson correlations between all feature pairs...")

    adj, abs_corr, mi_matrix = build_dependency_graph(X_train, feature_names, corr_threshold=0.10)

    # Show top 10 strongest feature relationships
    print("\n  Top 10 strongest feature dependencies:")
    print(f"  {'Feature A':<35} {'Feature B':<35} {'Strength':>8}")
    print("  " + "-"*80)

    pairs = []
    n = adj.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j] > 0 or adj[j, i] > 0:
                strength = max(adj[i, j], adj[j, i])
                pairs.append((feature_names[i], feature_names[j], strength))

    pairs.sort(key=lambda x: x[2], reverse=True)
    for fa, fb, strength in pairs[:10]:
        print(f"  {fa:<35} {fb:<35} {strength:>8.4f}")

    print(f"\n  Total edges in graph: {(adj > 0).sum()}")
    print("  (These are the relationships that get used during propagation)")

    # ── Step 4: Load model ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 4 — Load Trained Model")
    print("="*60)
    input_size = X_test_t.shape[1]
    model = MLP(input_size)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    print("  Loaded model.pth successfully")

    # ── Step 5: Run attacks and compare ──────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 5 — Comparing Standard vs Propagated Attack")
    print("="*60)

    # Baseline clean accuracy
    with torch.no_grad():
        preds_clean = model(X_test_t).argmax(dim=1)
    acc_clean = accuracy_score(y_test_t, preds_clean)
    print(f"\n  Baseline (clean)           : {acc_clean:.4f}  ({acc_clean*100:.1f}%)")

    # Standard FGSM (no propagation)
    X_fgsm = fgsm_attack(model, X_test_t.clone(), y_test_t, epsilon=0.2)
    with torch.no_grad():
        preds_fgsm = model(X_fgsm).argmax(dim=1)
    acc_fgsm = accuracy_score(y_test_t, preds_fgsm)
    print(f"  Standard FGSM              : {acc_fgsm:.4f}  ({acc_fgsm*100:.1f}%)")

    # Standard PGD (no propagation)
    X_pgd = pgd_attack(model, X_test_t.clone(), y_test_t,
                       epsilon=0.2, alpha=0.02, iters=10)
    with torch.no_grad():
        preds_pgd = model(X_pgd).argmax(dim=1)
    acc_pgd = accuracy_score(y_test_t, preds_pgd)
    print(f"  Standard PGD               : {acc_pgd:.4f}  ({acc_pgd*100:.1f}%)")

    # Propagated FGSM (with dependency graph + constraints)
    X_prop_fgsm = fgsm_propagated(
        model, X_test_t.clone(), y_test_t,
        epsilon=0.5,
        adj=adj,
        feature_names=feature_names,
        propagation_strength=0.5
    )
    with torch.no_grad():
        preds_prop_fgsm = model(X_prop_fgsm).argmax(dim=1)
    acc_prop_fgsm = accuracy_score(y_test_t, preds_prop_fgsm)
    print(f"  Propagated FGSM            : {acc_prop_fgsm:.4f}  ({acc_prop_fgsm*100:.1f}%)")

    # Propagated PGD (with dependency graph + constraints)
    X_prop_pgd = pgd_propagated(
        model, X_test_t.clone(), y_test_t,
        epsilon=0.4, alpha=0.04, iters=20,
        adj=adj,
        feature_names=feature_names,
        propagation_strength=0.3
    )
    with torch.no_grad():
        preds_prop_pgd = model(X_prop_pgd).argmax(dim=1)
    acc_prop_pgd = accuracy_score(y_test_t, preds_prop_pgd)
    print(f"  Propagated PGD             : {acc_prop_pgd:.4f}  ({acc_prop_pgd*100:.1f}%)")

    # Summary table
    print("\n  " + "-"*55)
    print(f"  {'Scenario':<30} {'Accuracy':>10} {'Drop':>10}")
    print("  " + "-"*55)
    print(f"  {'Baseline (clean)':<30} {acc_clean:>10.1%} {'—':>10}")
    print(f"  {'Standard FGSM':<30} {acc_fgsm:>10.1%} {(acc_clean-acc_fgsm)*100:>9.1f}pp")
    print(f"  {'Standard PGD':<30} {acc_pgd:>10.1%} {(acc_clean-acc_pgd)*100:>9.1f}pp")
    print(f"  {'Propagated FGSM':<30} {acc_prop_fgsm:>10.1%} {(acc_clean-acc_prop_fgsm)*100:>9.1f}pp")
    print(f"  {'Propagated PGD':<30} {acc_prop_pgd:>10.1%} {(acc_clean-acc_prop_pgd)*100:>9.1f}pp")
    print("  " + "-"*55)

    # Store all results for plotting
    acc_prop = acc_prop_fgsm  # keep variable for feature change section below

    # ── Step 6: Show what actually changed in features ───────────────────────
    print("\n" + "="*60)
    print("STEP 6 — What Changed in the Features (sample of 1 record)")
    print("="*60)

    sample_idx = 0
    original   = X_test_t[sample_idx].numpy()
    after_fgsm = X_fgsm[sample_idx].detach().numpy()
    after_prop = X_prop_fgsm[sample_idx].detach().numpy()

    print(f"\n  {'Feature':<35} {'Original':>10} {'Std FGSM':>10} {'Prop FGSM':>10} {'Changed?':>10}")
    print("  " + "-"*80)

    # Show first 15 features
    for i in range(min(15, len(feature_names))):
        orig = original[i]
        fgsm_val = after_fgsm[i]
        prop_val = after_prop[i]
        changed = "YES" if abs(prop_val - orig) > 0.001 else "no"
        frozen_flag = " (frozen)" if build_constraint_mask(feature_names)[i] == 0 else ""
        print(f"  {feature_names[i]:<35} {orig:>10.4f} {fgsm_val:>10.4f} {prop_val:>10.4f} {changed:>10}{frozen_flag}")

    # ── Step 7: Plot ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 7 — Plotting Results")
    print("="*60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: accuracy comparison bar chart ──────────────────────────────────
    labels = ["Baseline\n(clean)", "Standard\nFGSM", "Standard\nPGD",
              "Propagated\nFGSM", "Propagated\nPGD"]
    values = [acc_clean, acc_fgsm, acc_pgd, acc_prop_fgsm, acc_prop_pgd]
    colors = ["#4C72B0", "#C44E52", "#C44E52", "#8172B3", "#8172B3"]

    ax = axes[0]
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="white")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Test Accuracy", fontsize=11)
    ax.set_title("Standard vs Propagated Attack\n(lower = stronger attack)", fontsize=11)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4C72B0", label="Baseline"),
        Patch(facecolor="#C44E52", label="Standard attacks"),
        Patch(facecolor="#8172B3", label="Propagated attacks"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    # ── Right: Top 10 feature pairs — simple and clean ───────────────────────
    ax2 = axes[1]

    # Use RAW correlation matrix (not normalized adj) for visualization
    with np.errstate(invalid="ignore", divide="ignore"):
        raw_corr = np.corrcoef(X_train.T)
        raw_corr = np.nan_to_num(raw_corr, nan=0.0)
    abs_corr = np.abs(raw_corr)
    np.fill_diagonal(abs_corr, 0.0)

    # Collect meaningful feature pairs (skip one-hot siblings)
    pairs = []
    seen = set()
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            strength = abs_corr[i, j]
            if strength < 0.05:
                continue
            # Skip pairs from same one-hot group (same base feature)
            base_i = feature_names[i].rsplit("_", 1)[0]
            base_j = feature_names[j].rsplit("_", 1)[0]
            if base_i == base_j:
                continue
            key = (min(i,j), max(i,j))
            if key in seen:
                continue
            seen.add(key)
            # Clean up names
            def clean(n):
                n = n.replace("marital-status_", "marital_")
                n = n.replace("native-country_", "country_")
                n = n.replace("relationship_", "rel_")
                n = n.replace("occupation_", "occ_")
                n = n.replace("workclass_", "work_")
                n = n.replace("education_", "edu_")
                return n[:20]
            pairs.append((clean(feature_names[i]), clean(feature_names[j]), strength))

    pairs.sort(key=lambda x: x[2], reverse=True)
    top = pairs[:10]

    y_labels = [f"{a}  ↔  {b}" for a, b, _ in top]
    strengths = [s for _, _, s in top]
    colors2 = ["#1F4E79" if s > 0.3 else "#378ADD" if s > 0.15 else "#85B7EB"
               for s in strengths]

    ax2.barh(y_labels[::-1], strengths[::-1], color=colors2[::-1],
             height=0.6, edgecolor="white")

    for i, (s, label) in enumerate(zip(strengths[::-1], y_labels[::-1])):
        ax2.text(s + 0.003, i, f"{s:.2f}", va="center", fontsize=9, fontweight="bold")

    ax2.set_xlabel("Correlation strength", fontsize=10)
    ax2.set_title("Top 10 feature relationships\nused in propagation", fontsize=11)
    ax2.set_xlim(0, max(strengths) * 1.25)
    ax2.tick_params(axis="y", labelsize=9)
    ax2.axvline(x=0.15, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.text(0.15, -0.8, "threshold", color="gray", fontsize=8, ha="center")

    plt.suptitle("Attack Propagation Analysis — Team Immortals",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig("propagation_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved to propagation_results.png")

    print("\n" + "="*60)
    print("ALL STEPS COMPLETE")
    print("="*60)
    print("  Propagation is working correctly.")
    print("  Next step: run adversarial_train.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
