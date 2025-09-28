#!/usr/bin/env python3
"""
graph_metrics.py

Compute global, node-level, and ROI-level graph metrics from a connectome
(edges.csv + neurons.csv). Saves CSVs and plots into an output directory.

Usage:
    python graph_metrics.py \
      --nodes_csv D:/python/NEUROLOGY/neurons.csv \
      --edges_csv D:/python/NEUROLOGY/edges.csv \
      --out_dir D:/python/NEUROLOGY/graph_metrics_output \
      --sample_nodes 0 \
      --betweenness_k 100 \
      --seed 42

Notes:
- For very large graphs you can use --sample_nodes or --sample_edges to analyze a smaller subgraph.
- Betweenness centrality is expensive; we compute an approximation with parameter --betweenness_k (set 0 to skip).
"""
import os
import argparse
import ast
import json
import random
from collections import defaultdict, Counter

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x: x  # fallback

# -------------------------
# Utilities
# -------------------------
def safe_literal_eval(val):
    if pd.isna(val):
        return None
    if isinstance(val, (list, dict)):
        return val
    try:
        return ast.literal_eval(val)
    except Exception:
        return None

def compute_primary_roi(roi_info_str, input_rois_str):
    # Try roiInfo dict first (counts). Else fall back to first inputRoi.
    roi_info = safe_literal_eval(roi_info_str)
    if isinstance(roi_info, dict) and roi_info:
        # compute sum(pre+post) per ROI and pick max
        sums = {}
        for roi, counts in roi_info.items():
            if not isinstance(counts, dict):
                continue
            s = 0
            s += counts.get('pre', 0) or 0
            s += counts.get('post', 0) or 0
            s += counts.get('downstream', 0) or 0
            sums[roi] = s
        if sums:
            return max(sums.items(), key=lambda kv: kv[1])[0]
    # fallback: inputRois (list)
    input_rois = safe_literal_eval(input_rois_str)
    if isinstance(input_rois, (list, tuple)) and input_rois:
        return input_rois[0]
    return "unknown"

def approx_avg_shortest_path_length(G, n_samples=100, directed=True, seed=0):
    """Approximate average shortest path length by sampling source nodes.
       Works on large graphs as an approximation."""
    random.seed(seed)
    nodes = list(G.nodes)
    if len(nodes) == 0:
        return None
    samples = min(n_samples, len(nodes))
    chosen = random.sample(nodes, samples)
    total = 0
    count = 0
    for u in chosen:
        # use single_source_shortest_path_length (unweighted)
        lengths = nx.single_source_shortest_path_length(G, u)
        # exclude self
        for v, d in lengths.items():
            if v == u:
                continue
            total += d
            count += 1
    return (total / count) if count > 0 else None

def approx_global_efficiency(G, n_samples=100, seed=0):
    """Approximate global efficiency: average of 1/d for node pairs (sampled)."""
    random.seed(seed)
    nodes = list(G.nodes)
    if len(nodes) < 2:
        return None
    samples = min(n_samples, len(nodes))
    chosen = random.sample(nodes, samples)
    total_eff = 0.0
    pairs = 0
    for u in chosen:
        lengths = nx.single_source_shortest_path_length(G, u)
        for v, d in lengths.items():
            if u == v:
                continue
            if d > 0:
                total_eff += 1.0 / d
                pairs += 1
    return (total_eff / pairs) if pairs > 0 else None

# -------------------------
# Main workflow
# -------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nodes_csv", required=True)
    p.add_argument("--edges_csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--sample_nodes", type=int, default=0,
                   help="If >0, analyze only an induced subgraph of this many nodes (first N).")
    p.add_argument("--sample_edges", type=int, default=0,
                   help="If >0, sample up to this many edges randomly from edges.csv.")
    p.add_argument("--betweenness_k", type=int, default=100,
                   help="If >0, approximate betweenness centrality with k random sources. Set 0 to skip.")
    p.add_argument("--sp_samples", type=int, default=200,
                   help="Samples for approximate shortest path / efficiency.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    print("[STEP] Loading CSVs...")
    nodes_df = pd.read_csv(args.nodes_csv)
    edges_df = pd.read_csv(args.edges_csv)

    # Optional edge sampling (fast test)
    if args.sample_edges and args.sample_edges > 0:
        print(f"[INFO] Sampling {args.sample_edges} edges for faster analysis...")
        edges_df = edges_df.sample(n=min(args.sample_edges, len(edges_df)), random_state=args.seed).reset_index(drop=True)

    # Ensure expected columns
    required_edge_cols = {'bodyId_pre', 'bodyId_post'}
    if not required_edge_cols.issubset(set(edges_df.columns)):
        raise ValueError(f"edges.csv missing required columns: {required_edge_cols} (found {list(edges_df.columns)})")

    if 'weight' not in edges_df.columns:
        edges_df['weight'] = 1

    # Build directed graph
    print("[STEP] Building NetworkX DiGraph...")
    G = nx.from_pandas_edgelist(edges_df, source='bodyId_pre', target='bodyId_post',
                                edge_attr='weight', create_using=nx.DiGraph())
    print("[INFO] Graph nodes:", G.number_of_nodes())
    print("[INFO] Graph edges:", G.number_of_edges())

    # Optional node sampling / induced subgraph
    if args.sample_nodes and args.sample_nodes > 0:
        nodes_list = list(G.nodes)[:args.sample_nodes]
        G = G.subgraph(nodes_list).copy()
        print(f"[INFO] Using induced subgraph with {len(G.nodes)} nodes and {G.number_of_edges()} edges")

    out = args.out_dir

    # -------------------------
    # Global metrics
    # -------------------------
    print("[STEP] Computing global metrics...")
    global_metrics = {}
    global_metrics['nodes'] = G.number_of_nodes()
    global_metrics['edges'] = G.number_of_edges()
    global_metrics['directed'] = True
    # density for directed graph
    global_metrics['density'] = nx.density(G)
    # weakly connected components (directed graphs)
    wcc = list(nx.weakly_connected_components(G))
    wcc_sizes = sorted([len(c) for c in wcc], reverse=True)
    global_metrics['n_weakly_connected_components'] = len(wcc_sizes)
    global_metrics['largest_wcc_size'] = wcc_sizes[0] if wcc_sizes else 0

    # clustering (use undirected clustering)
    try:
        und = G.to_undirected()
        global_metrics['average_clustering'] = nx.average_clustering(und)
    except Exception:
        global_metrics['average_clustering'] = None

    # approximate average shortest path length & global efficiency
    global_metrics['approx_avg_shortest_path'] = approx_avg_shortest_path_length(G, n_samples=args.sp_samples, seed=args.seed)
    global_metrics['approx_global_efficiency'] = approx_global_efficiency(G, n_samples=args.sp_samples, seed=args.seed)

    # save global metrics
    with open(os.path.join(out, "global_metrics.json"), "w") as f:
        json.dump(global_metrics, f, indent=2)
    print("[DONE] Saved global_metrics.json")

    # -------------------------
    # Node-level metrics
    # -------------------------
    print("[STEP] Computing node-level metrics (degrees / strengths / pagerank)...")
    nodes = list(G.nodes)
    # degrees
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    deg = {n: in_deg.get(n, 0) + out_deg.get(n, 0) for n in nodes}
    # weighted degrees (strength)
    in_strength = dict(G.in_degree(weight='weight'))
    out_strength = dict(G.out_degree(weight='weight'))
    strength = {n: in_strength.get(n, 0) + out_strength.get(n, 0) for n in nodes}
    # clustering (undirected)
    clustering = {}
    try:
        clustering = nx.clustering(und)
    except Exception:
        clustering = {n: None for n in nodes}

    # pagerank
    try:
        pagerank = nx.pagerank(G, weight='weight')
    except Exception:
        pagerank = {n: 0.0 for n in nodes}

    # approximate betweenness (sampling)
    betweenness = {}
    if args.betweenness_k and args.betweenness_k > 0:
        print(f"[INFO] Computing approximate betweenness centrality with k={args.betweenness_k} sources...")
        betweenness = nx.betweenness_centrality(G, k=args.betweenness_k, normalized=True, weight=None, seed=args.seed)
    else:
        print("[INFO] Skipping betweenness centrality (set --betweenness_k > 0 to compute)")

    # assemble node metrics DataFrame
    node_rows = []
    # create mapping from bodyId to primary ROI using neurons.csv
    print("[INFO] Building ROI map from neurons.csv...")
    roi_map = {}
    for idx, r in nodes_df.iterrows():
        body = r.get('bodyId')
        primary = compute_primary_roi(r.get('roiInfo'), r.get('inputRois'))
        roi_map[body] = primary

    for n in tqdm(nodes):
        node_rows.append({
            'bodyId': n,
            'in_degree': in_deg.get(n, 0),
            'out_degree': out_deg.get(n, 0),
            'degree': deg.get(n, 0),
            'in_strength': in_strength.get(n, 0),
            'out_strength': out_strength.get(n, 0),
            'strength': strength.get(n, 0),
            'clustering': clustering.get(n, None),
            'pagerank': pagerank.get(n, 0.0),
            'betweenness': betweenness.get(n, None) if betweenness else None,
            'primary_roi': roi_map.get(n, 'unknown')
        })

    node_metrics_df = pd.DataFrame(node_rows)
    node_metrics_df.to_csv(os.path.join(out, "node_metrics.csv"), index=False)
    print("[DONE] Saved node_metrics.csv")

    # -------------------------
    # ROI-level aggregation
    # -------------------------
    print("[STEP] Aggregating ROI-level metrics...")
    roi_group = node_metrics_df.groupby('primary_roi').agg({
        'bodyId': 'count',
        'degree': ['mean', 'std', 'sum'],
        'strength': ['mean', 'std', 'sum'],
        'pagerank': 'mean',
        'clustering': 'mean'
    })
    # flatten columns
    roi_group.columns = ['_'.join(col).strip() for col in roi_group.columns.values]
    roi_group = roi_group.reset_index().rename(columns={'bodyId_count': 'n_neurons'})
    roi_group.to_csv(os.path.join(out, "roi_metrics.csv"), index=False)
    print("[DONE] Saved roi_metrics.csv")

    # -------------------------
    # Top hubs and plots
    # -------------------------
    print("[STEP] Generating plots...")
    # Degree histogram (log scale)
    plt.figure(figsize=(6, 4))
    degs = node_metrics_df['degree'].values
    plt.hist(degs, bins=100)
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.title("Degree distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "degree_hist.png"))
    plt.close()

    # Top 20 nodes by degree
    top_deg = node_metrics_df.sort_values('degree', ascending=False).head(20)
    plt.figure(figsize=(8, 5))
    plt.barh(top_deg['bodyId'].astype(str), top_deg['degree'])
    plt.gca().invert_yaxis()
    plt.xlabel("Degree")
    plt.title("Top 20 nodes by degree")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "top20_degree.png"))
    plt.close()

    # ROI: mean degree top 20
    top_roi = roi_group.sort_values('degree_mean', ascending=False).head(20)
    plt.figure(figsize=(10, 6))
    plt.barh(top_roi['primary_roi'].astype(str), top_roi['degree_mean'])
    plt.gca().invert_yaxis()
    plt.xlabel("Mean degree")
    plt.title("Top 20 ROIs by mean degree")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "top20_roi_mean_degree.png"))
    plt.close()

    # scatter pagerank vs degree (sample for readability)
    sample_df = node_metrics_df.sample(n=min(5000, len(node_metrics_df)), random_state=args.seed)
    plt.figure(figsize=(6, 6))
    plt.scatter(sample_df['degree'], sample_df['pagerank'], s=5, alpha=0.6)
    plt.xlabel("Degree")
    plt.ylabel("PageRank")
    plt.title("PageRank vs Degree (sample)")
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(out, "pagerank_vs_degree.png"))
    plt.close()

    # save global summary as text
    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write("Global metrics:\n")
        for k, v in global_metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\nTop 10 ROIs by mean degree:\n")
        for i, row in top_roi.iterrows():
            f.write(f"{row['primary_roi']}: mean_degree={row['degree_mean']}, n_neurons={row['n_neurons']}\n")

    print("[DONE] All outputs saved to", out)
    print("[INFO] Node metrics sample:")
    print(node_metrics_df.head())

if __name__ == "__main__":
    main()
