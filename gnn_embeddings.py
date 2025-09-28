import pandas as pd
import networkx as nx
import torch
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# ----------------------------
# GNN Encoder definition
# ----------------------------
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# ----------------------------
# Main workflow
# ----------------------------
def main():
    # 1️⃣ Load edges and neurons
    edges = pd.read_csv("D:/python/NEUROLOGY/edges.csv")
    neurons = pd.read_csv("D:/python/NEUROLOGY/neurons.csv")

    print("[INFO] Building NetworkX graph...")
    G = nx.from_pandas_edgelist(edges, 'bodyId_pre', 'bodyId_post', edge_attr='weight', create_using=nx.DiGraph)
    print(f"[INFO] Full graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # 2️⃣ Subsample ~5000 nodes for embedding
    sub_nodes = list(G.nodes())[:5000]
    H = G.subgraph(sub_nodes).copy()
    print(f"[INFO] Subgraph: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")

    # 3️⃣ Add simple node features (degree)
    for n in H.nodes():
        H.nodes[n]['degree'] = G.degree(n)

    # 4️⃣ Convert to PyG format
    data = from_networkx(H, group_node_attrs=['degree'])
    data.x = data.x.float()

    # 5️⃣ Train GNN encoder
    model = GCNEncoder(in_channels=data.x.shape[1], hidden_channels=64, out_channels=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = torch.norm(out)  # dummy unsupervised loss
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"[EPOCH {epoch}] Loss: {loss.item():.4f}")

    embeddings = out.detach().numpy()
    pd.DataFrame(embeddings).to_csv("D:/python/NEUROLOGY/embeddings.csv", index=False)
    print("[DONE] Saved embeddings.csv ✅")

    # 6️⃣ Functional clustering
    kmeans = KMeans(n_clusters=10, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    pd.DataFrame({"bodyId": sub_nodes, "cluster": labels}).to_csv("D:/python/NEUROLOGY/clusters.csv", index=False)
    print("[DONE] Saved clusters.csv ✅")

    # 7️⃣ Visualize embedding space
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    emb_2d = tsne.fit_transform(embeddings)
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap='tab10', s=8)
    plt.title("Neuron Embedding Space (t-SNE)")
    plt.savefig("D:/python/NEUROLOGY/embedding_plot.png")
    plt.close()
    print("[DONE] Saved embedding_plot.png ✅")

if __name__ == "__main__":
    main()
