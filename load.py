import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# 1️⃣ Load the CSVs
neurons = pd.read_csv("D:/python/NEUROLOGY/neurons.csv")
edges = pd.read_csv("D:/python/NEUROLOGY/edges.csv")

print("[INFO] Neurons loaded:", neurons.shape)
print("[INFO] Edges loaded:", edges.shape)

# 2️⃣ Build a graph
G = nx.from_pandas_edgelist(edges, source='bodyId_pre', target='bodyId_post', edge_attr='weight')

print("[INFO] Graph nodes:", G.number_of_nodes())
print("[INFO] Graph edges:", G.number_of_edges())

# 3️⃣ Inspect a neuron
print("\n[Sample neuron info]")
print(neurons.head(5))

# 4️⃣ Visualize a small subgraph (e.g., 50 neurons)
sub_nodes = list(G.nodes)[:50]
H = G.subgraph(sub_nodes)
plt.figure(figsize=(10, 8))
nx.draw(H, with_labels=False, node_size=30, edge_color='gray')
plt.title("Partial Hemibrain Neuron Connectivity")
plt.show()
