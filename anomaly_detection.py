import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

# 1️⃣ Load embeddings and clusters
embeddings = pd.read_csv("D:/python/NEUROLOGY/embeddings.csv")
clusters = pd.read_csv("D:/python/NEUROLOGY/clusters.csv")

# Combine them
data = embeddings.copy()
data["cluster"] = clusters["cluster"]

# 2️⃣ Train anomaly detector
print("[INFO] Training Isolation Forest for anomaly detection...")
clf = IsolationForest(contamination=0.05, random_state=42)
clf.fit(data.drop(columns=["cluster"]))
scores = clf.decision_function(data.drop(columns=["cluster"]))
labels = clf.predict(data.drop(columns=["cluster"]))  # -1 = anomaly, 1 = normal

data["anomaly_score"] = scores
data["anomaly_label"] = labels

# 3️⃣ Save results
data.to_csv("D:/python/NEUROLOGY/anomalies.csv", index=False)
print("[DONE] Saved anomalies.csv ✅")

# 4️⃣ Analyze anomaly distribution
summary = data.groupby("cluster")["anomaly_label"].apply(lambda x: (x == -1).sum())
summary = summary.reset_index().rename(columns={"anomaly_label": "anomalous_neurons"})
summary.to_csv("D:/python/NEUROLOGY/anomaly_summary.csv", index=False)
print("[DONE] Saved anomaly_summary.csv ✅")

# 5️⃣ Visualize anomaly space
plt.figure(figsize=(8, 6))
sns.histplot(data["anomaly_score"], kde=True)
plt.title("Anomaly Score Distribution")
plt.xlabel("Anomaly Score")
plt.savefig("D:/python/NEUROLOGY/anomaly_plot.png")
plt.close()
print("[DONE] Saved anomaly_plot.png ✅")

print("\n[INFO] Top 5 clusters with most anomalies:")
print(summary.sort_values("anomalous_neurons", ascending=False).head())
