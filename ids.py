import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay)

# data 
BASE = "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/"
FILES = ["KDDTrain+.txt", "KDDTest+.txt"]
for f in FILES:
    if not os.path.exists(f):
        try:
            import requests
            r = requests.get(BASE + f.replace("+", "%2B"), timeout=60)
            r.raise_for_status()
            open(f, "wb").write(r.content)
            print("Downloaded", f)
        except Exception as e:
            raise SystemExit(f"Could not download {f} ({e}). Download it manually and put it next to ids.py")

cols = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
 "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
 "root_shell","su_attempted","num_root","num_file_creations","num_shells",
 "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count",
 "srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
 "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
 "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
 "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
 "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
 "label","difficulty"]

train = pd.read_csv("KDDTrain+.txt", names=cols)
test = pd.read_csv("KDDTest+.txt", names=cols)

y_train = (train["label"] != "normal").astype(int).values   # 1 = attack
y_test = (test["label"] != "normal").astype(int).values
X_train = train.drop(columns=["label", "difficulty"])
X_test = test.drop(columns=["label", "difficulty"])

# attacks in the test set that never appear in training = "novel" attacks
train_attacks = set(train["label"].unique()) - {"normal"}
is_attack = y_test == 1
novel = is_attack & ~test["label"].isin(train_attacks).values
known = is_attack & ~novel
print(f"Test attacks: {is_attack.sum()} | known types: {known.sum()} | novel types: {novel.sum()}")

# pre processing 
cat = ["protocol_type", "service", "flag"]
num = [c for c in X_train.columns if c not in cat]
pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ("num", StandardScaler(), num),
])
Xtr = pre.fit_transform(X_train)
Xte = pre.transform(X_test)

# models 
models = {}

lr = LogisticRegression(max_iter=1000)
lr.fit(Xtr, y_train)
models["Logistic Regression"] = lr.predict(Xte)

rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(Xtr, y_train)
models["Random Forest"] = rf.predict(Xte)

# isolation Forest: trained on NORMAL traffic only (no attack labels used)
iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
iso.fit(Xtr[y_train == 0])
models["Isolation Forest"] = (iso.predict(Xte) == -1).astype(int)

# evaluation
rows = []
for name, pred in models.items():
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    rows.append({
        "Model": name,
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "False Positive Rate": fp / (fp + tn),
        "Recall (known attacks)": pred[known].mean(),
        "Recall (novel attacks)": pred[novel].mean(),
    })
results = pd.DataFrame(rows).round(3)
print("\n", results.to_string(index=False))
results.to_csv("results.csv", index=False)

# figures 
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (name, pred) in zip(axes, models.items()):
    ConfusionMatrixDisplay(confusion_matrix(y_test, pred),
                           display_labels=["Normal", "Attack"]).plot(ax=ax, colorbar=False)
    ax.set_title(name)
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=200)

x = np.arange(len(results))
plt.figure(figsize=(7, 4))
plt.bar(x - 0.2, results["Recall (known attacks)"], 0.4, label="Known attacks")
plt.bar(x + 0.2, results["Recall (novel attacks)"], 0.4, label="Novel attacks")
plt.xticks(x, results["Model"])
plt.ylabel("Recall")
plt.title("Detection rate: known vs unseen attack types")
plt.legend()
plt.tight_layout()
plt.savefig("known_vs_novel.png", dpi=200)
print("\nSaved results.csv, confusion_matrices.png, known_vs_novel.png")