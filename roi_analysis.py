import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1️⃣ Load data
neurons = pd.read_csv("D:/python/NEUROLOGY/neurons.csv")
anomalies = pd.read_csv("D:/python/NEUROLOGY/anomalies.csv")
clusters = pd.read_csv("D:/python/NEUROLOGY/clusters.csv")

# Merge all
anomalies["cluster"] = clusters["cluster"]
anomalies["bodyId"] = neurons["bodyId"][:len(anomalies)]
merged = anomalies.merge(neurons[["bodyId", "roiInfo"]], on="bodyId", how="left")

# Extract main ROI name
merged["primary_roi"] = merged["roiInfo"].astype(str).str.extract(r"'([A-Z]+\(R\)|[A-Z]+\(L\)|[A-Z]+)'")

# 2️⃣ ROI-level anomaly stats
roi_summary = merged.groupby("primary_roi")["anomaly_label"].apply(lambda x: (x == -1).sum())
roi_summary = roi_summary.sort_values(ascending=False).reset_index().rename(columns={"anomaly_label": "anomaly_count"})
roi_summary.to_csv("D:/python/NEUROLOGY/roi_anomaly_summary.csv", index=False)

print("[INFO] Top 10 ROIs with highest anomaly counts:")
print(roi_summary.head(10))

# 3️⃣ Visualization
plt.figure(figsize=(12, 6))
sns.barplot(data=roi_summary.head(15), x="primary_roi", y="anomaly_count", hue="primary_roi", palette="rocket", legend=False)
plt.xticks(rotation=45)
plt.title("Top Brain Regions by Anomaly Count")
plt.ylabel("Anomalous Neurons")
plt.xlabel("Brain Region (ROI)")
plt.tight_layout()
plt.savefig("D:/python/NEUROLOGY/roi_anomaly_plot.png")
plt.close()

print("[DONE] Saved roi_anomaly_summary.csv and roi_anomaly_plot.png ✅")
