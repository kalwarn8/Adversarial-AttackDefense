"""
Defense strategy — mix 3 types per batch:
  50% clean examples
  25% standard FGSM (epsilon=0.2)
  25% propagated FGSM (epsilon=0.3)

Using both standard and propagated examples prevents the model
from overfitting to one specific attack pattern, which caused
the 99% bug when only propagated examples were used.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

from preprocessing.preprocessing import load_data
from models.model import MLP
from attacks.FGSM import fgsm_attack
from attacks.propagation import build_dependency_graph, fgsm_propagated
from attacks.constraints import print_constraint_summary


# ── Config ────────────────────────────────────────────────────────────────────
EPSILON_STD   = 0.2   # standard FGSM epsilon (same as evaluate.py)
EPSILON_PROP  = 0.3   # propagated FGSM training epsilon
PROP_STRENGTH = 0.5
EPOCHS        = 200
LR            = 0.001
BATCH_SIZE    = 512


def main():
    print("\n" + "="*60)
    print("ADVERSARIAL TRAINING — TEAM IMMORTALS")
    print("="*60)

    # ── Load data ─────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, feature_names = load_data("data/adult.data")
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.long)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
    y_test_t  = torch.tensor(y_test.values, dtype=torch.long)

    print_constraint_summary()

    # ── Build dependency graph ────────────────────────────────────────────────
    print("[Graph] Building feature dependency graph...")
    adj, abs_corr, mi_matrix = build_dependency_graph(
        X_train, feature_names, corr_threshold=0.10
    )

    # ── Load baseline model as starting point ─────────────────────────────────
    input_size = X_train_t.shape[1]
    model = MLP(input_size)
    try:
        model.load_state_dict(torch.load("model.pth"))
        print("[Model] Warm start from model.pth")
    except FileNotFoundError:
        print("[Model] Training from scratch")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    n_train   = X_train_t.shape[0]

    print(f"\n[Train] Mix: 50% clean + 25% std-FGSM + 25% prop-FGSM")
    print(f"        Epochs={EPOCHS}, Batch={BATCH_SIZE}")
    print(f"        Training eps: std={EPSILON_STD}, prop={EPSILON_PROP}")
    print(f"        Evaluation eps: 0.5 (stronger than training)")
    print("-"*60)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(EPOCHS):
        model.train()

        perm   = torch.randperm(n_train)
        X_shuf = X_train_t[perm]
        y_shuf = y_train_t[perm]

        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, n_train, BATCH_SIZE):
            X_batch = X_shuf[start: start + BATCH_SIZE]
            y_batch = y_shuf[start: start + BATCH_SIZE]

            n        = len(X_batch)
            n_clean  = n // 2
            n_std    = n // 4
            # rest goes to propagated

            X_clean   = X_batch[:n_clean]
            y_clean   = y_batch[:n_clean]
            X_std_src = X_batch[n_clean: n_clean + n_std]
            y_std_src = y_batch[n_clean: n_clean + n_std]
            X_prp_src = X_batch[n_clean + n_std:]
            y_prp_src = y_batch[n_clean + n_std:]

            model.eval()

            # Standard FGSM examples
            X_std_adv = fgsm_attack(
                model, X_std_src.clone(), y_std_src,
                epsilon=EPSILON_STD
            )

            # Propagated FGSM examples
            X_prp_adv = fgsm_propagated(
                model, X_prp_src, y_prp_src,
                epsilon=EPSILON_PROP,
                adj=adj,
                feature_names=feature_names,
                propagation_strength=PROP_STRENGTH
            )

            model.train()

            # Combine all three types
            X_combined = torch.cat([X_clean, X_std_adv, X_prp_adv], dim=0)
            y_combined = torch.cat([y_clean, y_std_src,  y_prp_src],  dim=0)

            outputs = model(X_combined)
            loss    = criterion(outputs, y_combined)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:>3}/{EPOCHS}  loss={epoch_loss/n_batches:.4f}")

    # ── Evaluate robust model ─────────────────────────────────────────────────
    model.eval()
    print("\n" + "="*60)
    print("ROBUST MODEL EVALUATION")
    print("="*60)

    # Clean accuracy
    with torch.no_grad():
        clean_acc = accuracy_score(y_test_t, model(X_test_t).argmax(dim=1))
    print(f"  Clean accuracy                    : {clean_acc:.1%}")

    # Standard FGSM
    X_fgsm = fgsm_attack(model, X_test_t.clone(), y_test_t, epsilon=0.2)
    with torch.no_grad():
        fgsm_acc = accuracy_score(y_test_t, model(X_fgsm).argmax(dim=1))
    print(f"  Robust vs Std-FGSM (eps=0.2)      : {fgsm_acc:.1%}")

    # Propagated FGSM at eps=0.5 (stronger than training eps=0.3)
    X_prop = fgsm_propagated(
        model, X_test_t, y_test_t,
        epsilon=0.5, adj=adj,
        feature_names=feature_names,
        propagation_strength=0.5
    )
    with torch.no_grad():
        prop_acc = accuracy_score(y_test_t, model(X_prop).argmax(dim=1))
    print(f"  Robust vs Prop-FGSM (eps=0.5)     : {prop_acc:.1%}")
    print(f"\n  Recovery: 9% -> {prop_acc:.1%} (+{(prop_acc-0.09)*100:.1f}pp)")
    print(f"  Clean cost: 84% -> {clean_acc:.1%} ({(clean_acc-0.84)*100:+.1f}pp)")

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), "model_robust.pth")
    print("\n[Save] model_robust.pth saved")
    print("="*60)
    print("\nNext: python evaluate.py\n")


if __name__ == "__main__":
    main()
