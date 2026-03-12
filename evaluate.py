import torch
from sklearn.metrics import accuracy_score

from preprocessing.preprocessing import load_data
from models.model import MLP
from attacks.FGSM import fgsm_attack
from attacks.PGD import pgd_attack
import matplotlib.pyplot as plt


X_train, X_test, y_train, y_test = load_data("data/adult.data")

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.long)

input_size = X_test.shape[1]

model = MLP(input_size)
model.load_state_dict(torch.load("model.pth"))

# ---------- BASELINE ----------
model.eval()

outputs = model(X_test)
pred = outputs.argmax(dim=1)

baseline_acc = accuracy_score(y_test, pred)

print("Baseline Accuracy:", baseline_acc)


# ---------- FGSM ATTACK ----------
model.train()   # allow gradients

X_fgsm = fgsm_attack(model, X_test.clone().detach(), y_test)

outputs = model(X_fgsm)
pred = outputs.argmax(dim=1)

fgsm_acc = accuracy_score(y_test, pred)

print("FGSM Accuracy:", fgsm_acc)


# ---------- PGD ATTACK ----------
X_pgd = pgd_attack(model, X_test.clone().detach(), y_test)

outputs = model(X_pgd)
pred = outputs.argmax(dim=1)

pgd_acc = accuracy_score(y_test, pred)

print("PGD Accuracy:", pgd_acc)

# accuracies
labels = ["Baseline", "FGSM Attack", "PGD Attack"]
values = [baseline_acc, fgsm_acc, pgd_acc]

plt.figure(figsize=(6,4))

bars = plt.bar(labels, values)

plt.ylabel("Accuracy")
plt.title("Model Accuracy Under Adversarial Attacks")

# show values above bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height,
             f"{height:.2f}", ha='center', va='bottom')

plt.ylim(0,1)

plt.show()