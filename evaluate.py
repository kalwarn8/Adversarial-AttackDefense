"""
Full evaluation showing all scenarios:

  1. Baseline clean
  2. Standard FGSM
  3. Standard PGD
  4. Propagated FGSM  (our realistic attack)
  5. Propagated PGD   (our strongest attack)
  6. Robust clean     (after adversarial training)
  7. Robust + Prop-FGSM (defense works)
  8. Robust + Prop-PGD  (defense vs strongest)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import accuracy_score

from preprocessing.preprocessing import load_data
from models.model import MLP
from attacks.FGSM import fgsm_attack
from attacks.PGD import pgd_attack
from attacks.propagation import build_dependency_graph, fgsm_propagated, pgd_propagated
from attacks.constraints import print_constraint_summary


# ── Load data ─────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test, feature_names = load_data("data/adult.data")

X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test.values, dtype=torch.long)
input_size = X_test_t.shape[1]

# ── Print constraint summary ──────────────────────────────────────────────────
print_constraint_summary()

# ── Build dependency graph ────────────────────────────────────────────────────
print("[Graph] Building feature dependency graph (correlation + mutual information)...")
adj, abs_corr, mi_matrix = build_dependency_graph(X_train, feature_names, corr_threshold=0.10)

# ── Load baseline model ───────────────────────────────────────────────────────
baseline = MLP(input_size)
baseline.load_state_dict(torch.load("model.pth"))
baseline.eval()
print("[Model] Baseline model loaded")

# ── Load robust model ─────────────────────────────────────────────────────────
try:
    robust = MLP(input_size)
    robust.load_state_dict(torch.load("model_robust.pth"))
    robust.eval()
    has_robust = True
    print("[Model] Robust model loaded")
except FileNotFoundError:
    has_robust = False
    print("[Model] model_robust.pth not found — run adversarial_train.py first")
    print("        Showing baseline-only results for now\n")

# ── Helper ────────────────────────────────────────────────────────────────────
def get_acc(model, X):
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
    return accuracy_score(y_test_t, preds)

# ── Run all scenarios ─────────────────────────────────────────────────────────
print("\n[Eval] Running all scenarios...")
print("-"*55)

results = {}

# 1. Baseline clean
key_clean = "Baseline\n(clean)"
results[key_clean] = get_acc(baseline, X_test_t)
print(f"  1. Baseline clean        : {results[key_clean]:.1%}")

# 2. Standard FGSM
key_fgsm = "Standard\nFGSM"
X_fgsm = fgsm_attack(baseline, X_test_t.clone(), y_test_t, epsilon=0.2)
results[key_fgsm] = get_acc(baseline, X_fgsm)
print(f"  2. Standard FGSM         : {results[key_fgsm]:.1%}")

# 3. Standard PGD
key_pgd = "Standard\nPGD"
X_pgd = pgd_attack(baseline, X_test_t.clone(), y_test_t,
                   epsilon=0.2, alpha=0.02, iters=10)
results[key_pgd] = get_acc(baseline, X_pgd)
print(f"  3. Standard PGD          : {results[key_pgd]:.1%}")

# 4. Propagated FGSM — our realistic attack
key_pfgsm = "Prop-FGSM\n(realistic)"
X_prop_fgsm = fgsm_propagated(
    baseline, X_test_t.clone(), y_test_t,
    epsilon=0.5, adj=adj,
    feature_names=feature_names,
    propagation_strength=0.5
)
results[key_pfgsm] = get_acc(baseline, X_prop_fgsm)
print(f"  4. Propagated FGSM       : {results[key_pfgsm]:.1%}  <- our attack")

# 5. Propagated PGD — our strongest attack
key_ppgd = "Prop-PGD\n(strongest)"
X_prop_pgd = pgd_propagated(
    baseline, X_test_t.clone(), y_test_t,
    epsilon=0.4, alpha=0.04, iters=20,
    adj=adj, feature_names=feature_names,
    propagation_strength=0.3
)
results[key_ppgd] = get_acc(baseline, X_prop_pgd)
print(f"  5. Propagated PGD        : {results[key_ppgd]:.1%}  <- strongest attack")

if has_robust:
    # 6. Robust model clean
    key_rclean = "Robust\n(clean)"
    results[key_rclean] = get_acc(robust, X_test_t)
    print(f"  6. Robust clean          : {results[key_rclean]:.1%}")

    # 7. Robust vs Propagated FGSM
    key_rfgsm = "Robust +\nProp-FGSM"
    X_prop_fgsm_r = fgsm_propagated(
        robust, X_test_t.clone(), y_test_t,
        epsilon=0.5, adj=adj,
        feature_names=feature_names,
        propagation_strength=0.5
    )
    results[key_rfgsm] = get_acc(robust, X_prop_fgsm_r)
    print(f"  7. Robust + Prop-FGSM    : {results[key_rfgsm]:.1%}  <- defense")

    # 8. Robust vs Propagated PGD
    key_rpgd = "Robust +\nProp-PGD"
    X_prop_pgd_r = pgd_propagated(
        robust, X_test_t.clone(), y_test_t,
        epsilon=0.4, alpha=0.04, iters=20,
        adj=adj, feature_names=feature_names,
        propagation_strength=0.3
    )
    results[key_rpgd] = get_acc(robust, X_prop_pgd_r)
    print(f"  8. Robust + Prop-PGD     : {results[key_rpgd]:.1%}  <- defense vs strongest")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("RESULTS SUMMARY")
print("="*55)
baseline_acc = results.get(key_clean, 0)
for scenario, acc in results.items():
    label = scenario.replace("\n", " ")
    diff  = acc - baseline_acc
    diff_str = f"{diff:+.1%}" if diff != 0 else "baseline"
    print(f"  {label:<28} {acc:>7.1%}   {diff_str}")

if has_robust:
    prop_fgsm = results.get(key_pfgsm, 0)
    prop_pgd  = results.get(key_ppgd, 0)
    rob_fgsm  = results.get(key_rfgsm, 0)
    rob_pgd   = results.get(key_rpgd, 0)
    rob_clean = results.get(key_rclean, 0)
    print(f"\n  Key findings:")
    print(f"  Attack damage (Prop-FGSM)  : 84% -> {prop_fgsm:.1%}  ({(baseline_acc-prop_fgsm)*100:.1f}pp drop)")
    print(f"  Attack damage (Prop-PGD)   : 84% -> {prop_pgd:.1%}   ({(baseline_acc-prop_pgd)*100:.1f}pp drop)")
    print(f"  Defense recovery (FGSM)    : {prop_fgsm:.1%} -> {rob_fgsm:.1%}  ({(rob_fgsm-prop_fgsm)*100:.1f}pp recovery)")
    print(f"  Defense recovery (PGD)     : {prop_pgd:.1%} -> {rob_pgd:.1%}   ({(rob_pgd-prop_pgd)*100:.1f}pp recovery)")
    print(f"  Clean accuracy cost        : 84% -> {rob_clean:.1%}  ({(rob_clean-baseline_acc)*100:+.1f}pp)")

# ── Plot ──────────────────────────────────────────────────────────────────────
labels = list(results.keys())
values = list(results.values())

colors = []
for label in labels:
    if "Robust" in label:
        colors.append("#DD8452")   # orange = robust model
    elif "Prop" in label:
        colors.append("#8172B3")   # purple = propagated attacks
    else:
        colors.append("#4C72B0")   # blue = baseline/standard

fig, ax = plt.subplots(figsize=(13, 5))
bars = ax.bar(labels, values, color=colors, width=0.6, edgecolor="white")

for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.012,
            f"{h:.0%}", ha="center", va="bottom",
            fontsize=9, fontweight="bold")

ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.text(0.01, 0.52, "random chance (50%)",
        transform=ax.get_yaxis_transform(),
        color="gray", fontsize=8)

legend_elements = [
    Patch(color="#4C72B0", label="Baseline model — standard attacks"),
    Patch(color="#8172B3", label="Baseline model — propagated attacks (our novelty)"),
    Patch(color="#DD8452", label="Robust model — after adversarial training"),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

ax.set_ylabel("Test Accuracy", fontsize=11)
ax.set_ylim(0, 1.1)
ax.set_title(
    "Adversarial Robustness Evaluation — Team Immortals\n"
    "UCI Adult Income Dataset  |  Correlation + Mutual Information Dependency Graph",
    fontsize=11, fontweight="bold"
)

plt.tight_layout()
plt.savefig("results_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n[Plot] Saved to results_comparison.png")
