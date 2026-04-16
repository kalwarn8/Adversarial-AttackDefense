"""
Fairness Analysis — Demographic Parity Gap by Sex.

Metric: Demographic Parity Gap by Sex
  Gap = |P(predicted >50K | Male) - P(predicted >50K | Female)|
  Closer to 0 = more fair.

We measure this across 3 states:
  State 1 — Baseline model, clean inputs
  State 2 — Baseline model, under propagated attack
  State 3 — Robust model, clean inputs + under attack

"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from preprocessing.preprocessing import load_data
from models.model import MLP
from attacks.FGSM import fgsm_attack
from attacks.propagation import build_dependency_graph, fgsm_propagated


# ── Load sensitive attributes from raw test data ──────────────────────────────

def load_sensitive(test_path="data/adult.test"):
    """
    Reload raw test file to extract sex labels.
    These are dropped during encoding but needed for fairness analysis.
    """
    columns = [
        "age","workclass","fnlwgt","education","education-num",
        "marital-status","occupation","relationship","race","sex",
        "capital-gain","capital-loss","hours-per-week","native-country","income"
    ]
    df = pd.read_csv(test_path, names=columns, skiprows=1, na_values=" ?")
    df = df.dropna()
    df["income"] = df["income"].apply(lambda x: 1 if ">50K" in x else 0)
    # Strip whitespace from sex column
    sex = df["sex"].str.strip().values
    return sex, df["income"].values


# ── Demographic Parity Gap ────────────────────────────────────────────────────

def parity_gap(predictions, sex_labels):
    """
    Computes demographic parity gap between Male and Female.
    """
    # Strip all whitespace to handle any spacing variations
    sex_clean = np.array([s.strip() for s in sex_labels])
    
    mask_male   = sex_clean == "Male"
    mask_female = sex_clean == "Female"

    print(f"  [Debug] Male count: {mask_male.sum()}, Female count: {mask_female.sum()}")
    print(f"  [Debug] Unique sex values: {np.unique(sex_clean)}")

    rate_male   = predictions[mask_male].mean()   if mask_male.sum()   > 0 else 0.0
    rate_female = predictions[mask_female].mean() if mask_female.sum() > 0 else 0.0
    gap = abs(rate_male - rate_female)

    return gap, rate_male, rate_female


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*65)
    print("FAIRNESS ANALYSIS — DEMOGRAPHIC PARITY GAP BY SEX")
    print("="*65)
    print("Metric: Gap = |P(pred>50K | Male) - P(pred>50K | Female)|")
    print("Closer to 0 = more fair\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, feature_names = load_data("data/adult.data")
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.values, dtype=torch.long)

    # Load sensitive attributes
    sex, _ = load_sensitive("data/adult.test")

    # Align lengths (dropna may differ slightly)
    n = min(len(sex), len(X_test_t))
    sex      = sex[:n]
    X_test_t = X_test_t[:n]
    y_test_t = y_test_t[:n]
    y_np     = y_test_t.numpy()

    print(f"  Male samples   : {(sex == ' Male').sum()}")
    print(f"  Female samples : {(sex == ' Female').sum()}\n")

    # ── Build dependency graph ────────────────────────────────────────────────
    print("[Graph] Building dependency graph...")
    adj, _, _ = build_dependency_graph(X_train, feature_names, corr_threshold=0.10)

    input_size = X_test_t.shape[1]

    # ── Load models ───────────────────────────────────────────────────────────
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
        print("[Warning] model_robust.pth not found. Showing baseline only.\n")

    # ── Helper ────────────────────────────────────────────────────────────────
    def evaluate(model, X):
        with torch.no_grad():
            preds = model(X).argmax(dim=1).numpy()
        acc = accuracy_score(y_np, preds)
        gap, rm, rf = parity_gap(preds, sex)
        return acc, gap, rm, rf

    # ── Run all scenarios ─────────────────────────────────────────────────────
    results = []

    # State 1 — Baseline clean
    acc, gap, rm, rf = evaluate(baseline, X_test_t)
    results.append(("State 1", "Baseline (clean)", acc, gap, rm, rf))
    print(f"[State 1] Baseline clean")
    print(f"  Accuracy         : {acc:.1%}")
    print(f"  Male rate        : {rm:.1%}  (predicted >50K)")
    print(f"  Female rate      : {rf:.1%}  (predicted >50K)")
    print(f"  Parity gap       : {gap:.3f}\n")

    # State 2 — Baseline under standard FGSM
    X_fgsm = fgsm_attack(baseline, X_test_t.clone(), y_test_t, epsilon=0.2)
    acc, gap, rm, rf = evaluate(baseline, X_fgsm)
    results.append(("State 2a", "Baseline + Std FGSM", acc, gap, rm, rf))
    print(f"[State 2a] Baseline + Standard FGSM")
    print(f"  Accuracy         : {acc:.1%}")
    print(f"  Male rate        : {rm:.1%}")
    print(f"  Female rate      : {rf:.1%}")
    print(f"  Parity gap       : {gap:.3f}\n")

    # State 2 — Baseline under propagated FGSM
    X_prop = fgsm_propagated(
        baseline, X_test_t.clone(), y_test_t,
        epsilon=0.5, adj=adj,
        feature_names=feature_names,
        propagation_strength=0.5
    )
    acc, gap, rm, rf = evaluate(baseline, X_prop)
    results.append(("State 2b", "Baseline + Prop-FGSM", acc, gap, rm, rf))
    print(f"[State 2b] Baseline + Propagated FGSM (our attack)")
    print(f"  Accuracy         : {acc:.1%}")
    print(f"  Male rate        : {rm:.1%}")
    print(f"  Female rate      : {rf:.1%}")
    print(f"  Parity gap       : {gap:.3f}\n")

    if has_robust:
        # State 3 — Robust clean
        acc, gap, rm, rf = evaluate(robust, X_test_t)
        results.append(("State 3", "Robust (clean)", acc, gap, rm, rf))
        print(f"[State 3] Robust model (clean)")
        print(f"  Accuracy         : {acc:.1%}")
        print(f"  Male rate        : {rm:.1%}")
        print(f"  Female rate      : {rf:.1%}")
        print(f"  Parity gap       : {gap:.3f}\n")

        # State 3 — Robust under propagated FGSM
        X_prop_r = fgsm_propagated(
            robust, X_test_t.clone(), y_test_t,
            epsilon=0.5, adj=adj,
            feature_names=feature_names,
            propagation_strength=0.5
        )
        acc, gap, rm, rf = evaluate(robust, X_prop_r)
        results.append(("State 3+Atk", "Robust + Prop-FGSM", acc, gap, rm, rf))
        print(f"[State 3+Attack] Robust + Propagated FGSM")
        print(f"  Accuracy         : {acc:.1%}")
        print(f"  Male rate        : {rm:.1%}")
        print(f"  Female rate      : {rf:.1%}")
        print(f"  Parity gap       : {gap:.3f}\n")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("="*65)
    print("FAIRNESS SUMMARY TABLE")
    print("="*65)
    print(f"  {'Scenario':<28} {'Accuracy':>9} {'Gap':>8} {'Male':>8} {'Female':>8}")
    print("  " + "-"*65)
    for state, label, acc, gap, rm, rf in results:
        print(f"  {label:<28} {acc:>8.1%} {gap:>8.3f} {rm:>8.1%} {rf:>8.1%}")

    # ── Tradeoff discussion ───────────────────────────────────────────────────
    print("\n" + "="*65)
    print("FAIRNESS-ROBUSTNESS TRADEOFF DISCUSSION")
    print("="*65)

    baseline_gap = results[0][3]
    if has_robust:
        robust_gap = results[3][3]
        change     = robust_gap - baseline_gap

        print(f"\n  Baseline clean gap    : {baseline_gap:.3f}")
        print(f"  Robust clean gap      : {robust_gap:.3f}")
        print(f"  Change                : {change:+.3f}")

        if change < 0:
            print("\n  Finding: Adversarial training IMPROVED fairness slightly.")
            print("  The gap narrowed, meaning the robust model predicts high")
            print("  income more equally across male and female groups.")
            print("  This is a positive side-effect of the defense.")
        elif change > 0.02:
            print("\n  Finding: Adversarial training slightly WORSENED fairness.")
            print("  This is a known tradeoff — robust models can amplify")
            print("  existing dataset biases as they learn more aggressive")
            print("  decision boundaries. The change is small but worth noting.")
        else:
            print("\n  Finding: Adversarial training had minimal effect on fairness.")
            print("  The gap stayed approximately the same, suggesting the")
            print("  defense neither helps nor hurts demographic parity.")

    print("\n  Note: We report and discuss this tradeoff as scoped by the")
    print("="*65)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_fairness(results)


def plot_fairness(results):
    labels   = [r[1] for r in results]
    accs     = [r[2] for r in results]
    gaps     = [r[3] for r in results]
    males    = [r[4] for r in results]
    females  = [r[5] for r in results]

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: accuracy + parity gap side by side
    ax = axes[0]
    w = 0.35
    b1 = ax.bar(x - w/2, accs, w, label="Accuracy", color="#4C72B0", edgecolor="white")
    b2 = ax.bar(x + w/2, gaps, w, label="Parity gap (lower=fairer)", color="#C44E52", edgecolor="white")

    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f"{h:.0%}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Value", fontsize=10)
    ax.set_title("Accuracy vs Fairness Gap\nacross all scenarios", fontsize=10)
    ax.legend(fontsize=8)
    ax.axhline(y=0, color="gray", linewidth=0.5)

    # Right: male vs female positive prediction rates
    ax2 = axes[1]
    ax2.bar(x - w/2, males,   w, label="Male rate",   color="#4C72B0", edgecolor="white")
    ax2.bar(x + w/2, females, w, label="Female rate", color="#DD8452", edgecolor="white")

    for i, (m, f) in enumerate(zip(males, females)):
        ax2.text(i - w/2, m + 0.01, f"{m:.0%}", ha="center", va="bottom", fontsize=8)
        ax2.text(i + w/2, f + 0.01, f"{f:.0%}", ha="center", va="bottom", fontsize=8)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax2.set_ylim(0, 0.7)
    ax2.set_ylabel("Rate predicted >50K income", fontsize=10)
    ax2.set_title("Male vs Female Prediction Rates\n(closer together = fairer)", fontsize=10)
    ax2.legend(fontsize=8)

    plt.suptitle("Fairness-Robustness Tradeoff Analysis — Team Immortals\n"
                 "Metric: Demographic Parity Gap by Sex",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("fairness_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n[Plot] Saved to fairness_analysis.png")


if __name__ == "__main__":
    main()
