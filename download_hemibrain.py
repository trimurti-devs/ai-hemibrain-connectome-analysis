from neuprint import Client, fetch_neurons, fetch_adjacencies
import pandas as pd
import networkx as nx

# 🔑 Token
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImFiaGlzaGVrZ2hvc2hoZXRjNzEyMTAyQGdtYWlsLmNvbSIsImxldmVsIjoibm9hdXRoIiwiaW1hZ2UtdXJsIjoiaHR0cHM6Ly9saDMuZ29vZ2xldXNlcmNvbnRlbnQuY29tL2EvQUNnOG9jSXlWZWc4eTF5bi1YeG1uYmhwZjRta1Y2TEkxdVZUcERCdXljLU5Sc1NING1nLXRRPXM5Ni1jP3N6PTUwP3N6PTUwIiwiZXhwIjoxOTM4OTU3MDUzfQ.0aR4WDChwSdjHuBqtQ6bq5ZOhu3o0Df1qUeKDiEMGsY"

# Step 1: Connect
client = Client("https://neuprint.janelia.org", dataset="hemibrain:v1.2.1", token=TOKEN)
print("[INFO] Connected to neuPrint ✅")

# Step 2: Download neuron metadata
print("[INFO] Downloading neuron metadata...")
neurons, _ = fetch_neurons(None)
neurons.to_csv("D:/python/NEUROLOGY/neurons.csv", index=False)
print(f"[DONE] Saved neurons.csv ({neurons.shape[0]} neurons)")

# Step 3: Download real edges using Cypher query
print("[INFO] Fetching real synaptic connections...")

query = """
MATCH (a:Neuron)-[c:ConnectsTo]->(b:Neuron)
RETURN a.bodyId AS bodyId_pre, b.bodyId AS bodyId_post, c.weight AS weight
"""
edges = client.fetch_custom(query)
print(f"[INFO] Total edges downloaded: {edges.shape[0]}")

# ✅ Save to CSV
edges.to_csv("D:/python/NEUROLOGY/edges.csv", index=False)
print("[DONE] Saved edges.csv ✅")

# ✅ Preview
print("\n[Preview of neurons]")
print(neurons.head())

print("\n[Preview of edges]")
print(edges.head())
print("Edges columns:", list(edges.columns))

# Step 4: Build the NetworkX graph
print("[INFO] Building NetworkX graph...")
G = nx.from_pandas_edgelist(
    edges,
    source='bodyId_pre',
    target='bodyId_post',
    edge_attr='weight',
    create_using=nx.DiGraph
)
print(f"[INFO] Graph built ✅ | Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

nx.write_graphml(G, "D:/python/NEUROLOGY/hemibrain_graph.graphml")
print("[DONE] Graph saved as hemibrain_graph.graphml ✅")
