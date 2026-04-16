"""
Creates a clean 3-phase story chart:

  Phase 1 — Before Attack        : Baseline clean accuracy
  Phase 2 — After Propagation Attack : Accuracy under our realistic attacks
  Phase 3 — After Adversarial Training : Accuracy after defense

"""

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.metrics import accuracy_score

from preprocessing.preprocessing import load_data
from models.model import MLP
from attacks.propagation import build_dependency_graph, fgsm_propagated, pgd_propagated
from attacks.FGSM import fgsm_attack
from attacks.PGD import pgd_attack


# ── Load data ─────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test, feature_names = load_data("data/adult.data")
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test.values, dtype=torch.long)
input_size = X_test_t.shape[1]

# ── Build graph ───────────────────────────────────────────────────────────────
print("[Graph] Building dependency graph...")
adj, abs_corr, mi_matrix = build_dependency_graph(X_train, feature_names, corr_threshold=0.10)

# ── Load models ───────────────────────────────────────────────────────────────
def get_acc(model, X):
    model.eval()
    with torch.no_grad():
        return accuracy_score(y_test_t, model(X).argmax(dim=1))

baseline = MLP(input_size)
baseline.load_state_dict(torch.load("model.pth"))
baseline.eval()

try:
    robust = MLP(input_size)
    robust.load_state_dict(torch.load("model_robust.pth"))
    robust.eval()
    has_robust = True
except FileNotFoundError:
    has_robust = False
    print("[Warning] model_robust.pth not found. Run adversarial_train.py first.")

# ── Compute all accuracies ────────────────────────────────────────────────────
print("[Eval] Computing accuracies...")

# Phase 1 — Before attack
acc_clean = get_acc(baseline, X_test_t)

# Phase 2 — After propagation attacks (on baseline)
X_prop_fgsm = fgsm_propagated(
    baseline, X_test_t.clone(), y_test_t,
    epsilon=0.5, adj=adj, feature_names=feature_names, propagation_strength=0.5
)
acc_prop_fgsm = get_acc(baseline, X_prop_fgsm)

X_prop_pgd = pgd_propagated(
    baseline, X_test_t.clone(), y_test_t,
    epsilon=0.4, alpha=0.04, iters=20,
    adj=adj, feature_names=feature_names, propagation_strength=0.3
)
acc_prop_pgd = get_acc(baseline, X_prop_pgd)

# Also standard attacks for comparison
X_fgsm = fgsm_attack(baseline, X_test_t.clone(), y_test_t, epsilon=0.2)
acc_fgsm = get_acc(baseline, X_fgsm)

X_pgd = pgd_attack(baseline, X_test_t.clone(), y_test_t, epsilon=0.2, alpha=0.02, iters=10)
acc_pgd = get_acc(baseline, X_pgd)

# Phase 3 — After adversarial training (on robust model)
if has_robust:
    acc_robust_clean = get_acc(robust, X_test_t)

    X_prop_fgsm_r = fgsm_propagated(
        robust, X_test_t.clone(), y_test_t,
        epsilon=0.5, adj=adj, feature_names=feature_names, propagation_strength=0.5
    )
    acc_robust_prop_fgsm = get_acc(robust, X_prop_fgsm_r)

    X_prop_pgd_r = pgd_propagated(
        robust, X_test_t.clone(), y_test_t,
        epsilon=0.4, alpha=0.04, iters=20,
        adj=adj, feature_names=feature_names, propagation_strength=0.3
    )
    acc_robust_prop_pgd = get_acc(robust, X_prop_pgd_r)

# ── Print summary ─────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("RESULTS")
print("="*55)
print(f"  Phase 1 — Before attack")
print(f"    Baseline clean               : {acc_clean:.1%}")
print(f"\n  Phase 2 — After propagation attack (undefended)")
print(f"    Standard FGSM                : {acc_fgsm:.1%}")
print(f"    Standard PGD                 : {acc_pgd:.1%}")
print(f"    Propagated FGSM (realistic)  : {acc_prop_fgsm:.1%}")
print(f"    Propagated PGD  (strongest)  : {acc_prop_pgd:.1%}")
if has_robust:
    print(f"\n  Phase 3 — After adversarial training (defended)")
    print(f"    Robust clean                 : {acc_robust_clean:.1%}")
    print(f"    Robust + Prop-FGSM           : {acc_robust_prop_fgsm:.1%}")
    print(f"    Robust + Prop-PGD            : {acc_robust_prop_pgd:.1%}")
print("="*55)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))

# Group positions
x_phase1 = [0]
x_phase2 = [2, 3, 4, 5]
x_phase3 = [7, 8, 9] if has_robust else []

labels_p1 = ["Before attack\n(clean)"]
labels_p2 = ["Std FGSM\n(unrealistic)", "Std PGD\n(unrealistic)",
             "Prop-FGSM\n(realistic)", "Prop-PGD\n(strongest)"]
labels_p3 = ["Robust\n(clean)", "Robust +\nProp-FGSM", "Robust +\nProp-PGD"] if has_robust else []

values_p1 = [acc_clean]
values_p2 = [acc_fgsm, acc_pgd, acc_prop_fgsm, acc_prop_pgd]
values_p3 = [acc_robust_clean, acc_robust_prop_fgsm, acc_robust_prop_pgd] if has_robust else []

# Colors
colors_p1 = ["#4C72B0"]
colors_p2 = ["#E07070", "#E07070", "#8172B3", "#5A4A9A"]
colors_p3 = ["#2CA02C", "#2CA02C", "#2CA02C"] if has_robust else []

all_x      = x_phase1 + x_phase2 + x_phase3
all_labels = labels_p1 + labels_p2 + labels_p3
all_values = values_p1 + values_p2 + values_p3
all_colors = colors_p1 + colors_p2 + colors_p3

bars = ax.bar(all_x, all_values, color=all_colors, width=0.7, edgecolor="white", linewidth=0.8)

# Value labels
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.012,
            f"{h:.0%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

# Phase background shading
ax.axvspan(-0.5, 0.8,  alpha=0.05, color="#4C72B0", label="_nolegend_")
ax.axvspan(1.4,  5.6,  alpha=0.05, color="#C44E52", label="_nolegend_")
if has_robust:
    ax.axvspan(6.4, 9.6, alpha=0.05, color="#2CA02C", label="_nolegend_")

# Phase labels at top
ax.text(0,    1.06, "Phase 1\nBefore attack", ha="center", fontsize=10,
        fontweight="bold", color="#1F4E79")
ax.text(3.5,  1.06, "Phase 2\nAfter propagation attack", ha="center", fontsize=10,
        fontweight="bold", color="#8B0000")
if has_robust:
    ax.text(8, 1.06, "Phase 3\nAfter adversarial training", ha="center", fontsize=10,
            fontweight="bold", color="#1B5E20")

# Divider lines
ax.axvline(x=1.2,  color="gray", linestyle="--", linewidth=1, alpha=0.4)
if has_robust:
    ax.axvline(x=6.2, color="gray", linestyle="--", linewidth=1, alpha=0.4)

# Random chance line
ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
ax.text(0.01, 0.51, "random chance (50%)",
        transform=ax.get_yaxis_transform(), color="gray", fontsize=8)

# X axis labels
ax.set_xticks(all_x)
ax.set_xticklabels(all_labels, fontsize=8.5)

# Legend
legend_elements = [
    mpatches.Patch(color="#4C72B0", label="Baseline model (no attack)"),
    mpatches.Patch(color="#E07070", label="Standard attacks (unrealistic)"),
    mpatches.Patch(color="#8172B3", label="Propagated attacks (our novelty — realistic)"),
    mpatches.Patch(color="#2CA02C", label="After adversarial training (defense)"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=8.5)

ax.set_ylabel("Test Accuracy", fontsize=11)
ax.set_ylim(0, 1.18)
ax.set_xlim(-0.6, 9.8 if has_robust else 5.8)
ax.set_title(
    "Adversarial Robustness — Team Immortals\n"
    "Before Attack  →  After Propagation Attack  →  After Adversarial Training",
    fontsize=12, fontweight="bold"
)

plt.tight_layout()
plt.savefig("story_chart.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n[Plot] Saved to story_chart.png")
