"""
app.py — Simple Adversarial Robustness Demo
Team Immortals: Shaan Thakkar & Neel Kalwar

Run: python app.py → open http://localhost:5000
"""

from flask import Flask, jsonify, send_from_directory
import torch, numpy as np, os, sys
sys.path.insert(0, os.path.dirname(__file__))

from preprocessing.preprocessing import load_data
from models.model import MLP
from attacks.FGSM import fgsm_attack
from attacks.propagation import build_dependency_graph, fgsm_propagated
from attacks.PGD import pgd_attack
from attacks.propagation import pgd_propagated
from sklearn.metrics import accuracy_score

app = Flask(__name__, template_folder="templates")

# ── Boot: load everything once ────────────────────────────────────────────────
print("[Boot] Loading data and models...")
X_train, X_test, y_train, y_test, feature_names = load_data("data/adult.data")
X_test_t = torch.tensor(X_test,  dtype=torch.float32)
y_test_t = torch.tensor(y_test.values, dtype=torch.long)
input_size = X_test_t.shape[1]

baseline = MLP(input_size)
baseline.load_state_dict(torch.load("model.pth", map_location="cpu"))
baseline.eval()
print("[Boot] ✓ Baseline model")

robust = None
try:
    robust = MLP(input_size)
    robust.load_state_dict(torch.load("model_robust.pth", map_location="cpu"))
    robust.eval()
    print("[Boot] ✓ Robust model")
except:
    print("[Boot] ✗ model_robust.pth not found")

print("[Boot] Building dependency graph...")
adj, _, _ = build_dependency_graph(X_train, feature_names, corr_threshold=0.10)

def get_acc(model, X):
    with torch.no_grad():
        return round(accuracy_score(
            y_test_t.numpy(), model(X).argmax(dim=1).numpy()) * 100, 1)

def get_pred(model, x):
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)[0]
    pred = logits.argmax(dim=1).item()
    return pred, round(float(probs[pred]), 3)

# ── Precompute accuracy numbers on full test set ──────────────────────────────
print("[Boot] Computing accuracy numbers (takes ~2 min)...")
y = y_test_t.clone()

acc_baseline = get_acc(baseline, X_test_t)
print(f"  Baseline: {acc_baseline}%")

X_prop_fgsm = fgsm_propagated(baseline, X_test_t.clone(), y,
    epsilon=0.5, adj=adj, feature_names=feature_names, propagation_strength=0.5)
acc_attacked = get_acc(baseline, X_prop_fgsm)
print(f"  Attacked: {acc_attacked}%")

acc_defended = None
X_rob_fgsm   = None
if robust:
    X_rob_fgsm = fgsm_propagated(robust, X_test_t.clone(), y,
        epsilon=0.5, adj=adj, feature_names=feature_names, propagation_strength=0.5)
    acc_defended = get_acc(robust, X_rob_fgsm)
    print(f"  Defended: {acc_defended}%")

# ── Find a demo person: baseline says >50K, attack flips, defense recovers ────
DEMO_IDX = None
print("[Boot] Finding good demo sample...")
for i in range(len(X_test_t)):
    x = X_test_t[i:i+1]
    p_base, conf_base = get_pred(baseline, x)
    if p_base != 1 or conf_base < 0.80:
        continue
    x_adv = X_prop_fgsm[i:i+1]
    p_atk, _ = get_pred(baseline, x_adv)
    if p_atk != 0:
        continue
    if robust:
        x_rob = X_rob_fgsm[i:i+1]
        p_rob, _ = get_pred(robust, x_rob)
        # ideally defense recovers, but accept even if not
    DEMO_IDX = i
    break

if DEMO_IDX is None:
    DEMO_IDX = 0
    print("[Boot] Using fallback sample 0")
else:
    print(f"[Boot] Demo sample: index {DEMO_IDX}")

# ── Build person card from raw data ──────────────────────────────────────────
def build_person_card(idx):
    """
    Reconstruct a human-readable profile from the raw dataset row.
    Returns a dict with plain-English field values.
    """
    import pandas as pd
    columns = ["age","workclass","fnlwgt","education","education-num",
               "marital-status","occupation","relationship","race","sex",
               "capital-gain","capital-loss","hours-per-week","native-country","income"]
    df = pd.read_csv("data/adult.data", names=columns, na_values=" ?").dropna()
    df["income"] = df["income"].apply(lambda x: 1 if ">50K" in x else 0)

    # Match index accounting for preprocessing drop
    row = df.iloc[idx]

    edu_map = {
        1:"Preschool", 2:"1st–4th", 3:"5th–6th", 4:"7th–8th",
        5:"9th grade", 6:"10th grade", 7:"11th grade", 8:"12th grade",
        9:"High school grad", 10:"Some college", 11:"Assoc (vocational)",
        12:"Assoc (academic)", 13:"Bachelors", 14:"Masters",
        15:"Professional school", 16:"Doctorate"
    }

    return {
        "age":           int(row["age"]),
        "sex":           str(row["sex"]).strip(),
        "race":          str(row["race"]).strip(),
        "education":     edu_map.get(int(row["education-num"]), str(row["education"]).strip()),
        "occupation":    str(row["occupation"]).strip().replace("-", " "),
        "marital":       str(row["marital-status"]).strip().replace("-", " "),
        "hours":         int(row["hours-per-week"]),
        "capital_gain":  int(row["capital-gain"]),
        "workclass":     str(row["workclass"]).strip().replace("-", " "),
        "true_income":   int(row["income"]),
    }

person_card = build_person_card(DEMO_IDX)
print(f"[Boot] Person: {person_card['age']}yo {person_card['sex']}, {person_card['occupation']}")

# Get predictions for demo sample
x_demo   = X_test_t[DEMO_IDX:DEMO_IDX+1]
x_demo_adv = X_prop_fgsm[DEMO_IDX:DEMO_IDX+1]

p_clean, c_clean = get_pred(baseline, x_demo)
p_atk,   c_atk   = get_pred(baseline, x_demo_adv)
p_rob,   c_rob    = get_pred(robust, X_rob_fgsm[DEMO_IDX:DEMO_IDX+1]) if robust and X_rob_fgsm is not None else (None, None)

# What features changed
delta = (x_demo_adv - x_demo).detach().numpy()[0]
orig  = x_demo.detach().numpy()[0]
changes = []
for i, name in enumerate(feature_names):
    n = name.lower()
    frozen = any(k in n for k in ["sex_","race_","native-country"])
    d = float(delta[i])
    if abs(d) > 0.001 and not frozen:
        changes.append({"name": name, "delta": round(d,3), "up": d > 0})
changes.sort(key=lambda c: abs(c["delta"]), reverse=True)

# Map feature names to plain English
FEAT_MAP = {
    "age": ("Age", "👤"),
    "education-num": ("Education Level", "🎓"),
    "hours-per-week": ("Hours per Week", "⏰"),
    "capital-gain": ("Capital Gain", "💰"),
    "capital-loss": ("Capital Loss", "📉"),
    "fnlwgt": ("Census Weight", "📊"),
}
GROUP_MAP = {
    "workclass": ("Work Class", "🏢"),
    "education": ("Education Type", "🎓"),
    "marital-status": ("Marital Status", "💍"),
    "occupation": ("Occupation", "💼"),
    "relationship": ("Relationship", "👨‍👩‍👧"),
}

def plain_feat(name):
    if name in FEAT_MAP:
        return FEAT_MAP[name]
    for group, (label, icon) in GROUP_MAP.items():
        if name.startswith(group.replace("-","")):
            return (label, icon)
        if name.startswith(group):
            return (label, icon)
    clean = name.replace("_"," ").replace("-"," ").title()
    return (clean, "📌")

plain_changes = []
for c in changes[:8]:
    label, icon = plain_feat(c["name"])
    plain_changes.append({
        "label": label,
        "icon":  icon,
        "up":    c["up"],
        "dir":   "Increased ↑" if c["up"] else "Decreased ↓",
    })

print(f"[Boot] ✓ Ready at http://localhost:5000")
print(f"""
  ┌─────────────────────────────┐
  │  Baseline:  {acc_baseline}%           │
  │  Attacked:  {acc_attacked}%            │
  │  Defended:  {acc_defended}%           │
  └─────────────────────────────┘
""")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")

@app.route("/api/demo")
def demo():
    return jsonify({
        # Person card
        "person": person_card,

        # Per-person predictions
        "pred_clean":  {"label": ">$50K" if p_clean==1 else "≤$50K",
                        "income": p_clean==1, "conf": round(c_clean*100)},
        "pred_attack": {"label": ">$50K" if p_atk==1 else "≤$50K",
                        "income": p_atk==1,   "conf": round(c_atk*100)},
        "pred_robust": {"label": ">$50K" if p_rob==1 else "≤$50K",
                        "income": p_rob==1,   "conf": round(c_rob*100)} if p_rob is not None else None,

        # Accuracy bars (full test set)
        "acc_baseline": acc_baseline,
        "acc_attacked": acc_attacked,
        "acc_defended": acc_defended,
        "has_robust":   robust is not None,

        # What changed
        "changes": plain_changes,

        # Flipped?
        "flipped":   p_atk != p_clean,
        "recovered": p_rob == p_clean if p_rob is not None else None,
    })

@app.route("/api/status")
def status():
    return jsonify({"ok": True, "has_robust": robust is not None})

if __name__ == "__main__":
    app.run(debug=False, port=5000)
