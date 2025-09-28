# Main workflow for Neurology Thesis: Multimodal Connectome Analysis for AD Detection

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from gnn_embeddings import GCNEncoder, generate_node_embeddings
from disease_detection import GNNClassifier, train_classifier
import torch
from torch_geometric.utils import from_networkx

def main():
    # Step 1: Load existing hemibrain data (neuron-level)
    print("[INFO] Loading hemibrain data...")
    neurons = pd.read_csv("D:/python/NEUROLOGY/neurons.csv")
    edges = pd.read_csv("D:/python/NEUROLOGY/edges.csv")
    G = nx.from_pandas_edgelist(edges, source='bodyId_pre', target='bodyId_post', edge_attr='weight', create_using=nx.DiGraph)
    print(f"[INFO] Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Step 2: Subsample for GNN (hemibrain is too large)
    sub_nodes = list(G.nodes)[:1000]  # Subsample 1000 neurons
    H = G.subgraph(sub_nodes).copy()
    print(f"[INFO] Subgraph: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")

    # Step 2.5: Add node attributes from neurons.csv
    for node in H.nodes:
        neuron_data = neurons[neurons['bodyId'] == node]
        if not neuron_data.empty:
            H.nodes[node]['pre'] = neuron_data['pre'].values[0]
            H.nodes[node]['post'] = neuron_data['post'].values[0]
            # Add more if needed

    # Step 3: Convert to PyG
    pyg_data = from_networkx(H, group_node_attrs=['pre', 'post'])  # Use neuron features
    pyg_data.x = pyg_data.x.float()
    pyg_data.edge_attr = pyg_data.weight.float().unsqueeze(1)  # Edge weights

    # Step 4: Generate embeddings
    print("[INFO] Training GNN for embeddings...")
    model = GCNEncoder(in_channels=pyg_data.x.shape[1], hidden_channels=64, out_channels=32)
    # Placeholder training (in real, train with loss)
    embeddings = generate_node_embeddings(model, pyg_data)
    print(f"[INFO] Node embeddings shape: {embeddings.shape}")

    # Step 5: Visualize subgraph
    plt.figure(figsize=(10, 8))
    nx.draw(H, with_labels=False, node_size=10, edge_color='gray')
    plt.title("Hemibrain Neuron Subgraph")
    plt.savefig("hemibrain_subgraph.png")
    plt.close()

    # Step 6: Placeholder for disease detection (need labels)
    # If clinical data available, add classification
    print("[INFO] Workflow complete. For AD detection, integrate region-level data.")

if __name__ == "__main__":
    main()
