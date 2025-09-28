# AI-Driven Analysis of the Drosophila Hemibrain Connectome

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

This repository implements an AI pipeline for analyzing the Drosophila hemibrain connectome—a high-resolution synaptic dataset from the Janelia neuPrint API—using Graph Neural Networks (GNNs) for embeddings, anomaly detection, and Region of Interest (ROI) insights. The work serves as a proxy for neurodegenerative disease modeling (e.g., Alzheimer's and ASD) and is detailed in the accompanying thesis paper [`thesis_paper.md`](thesis_paper.md).

## Overview
The hemibrain dataset is a real-world, high-resolution map of synaptic connections in half the fruit fly (Drosophila) brain, containing approximately 186,000 neurons (nodes) and over 7 million synapses (directed edges). This forms a large-scale directed graph that mimics brain wiring, making it perfect for applying AI techniques in neurobiology. Our project provides a beginner-friendly, modular Python pipeline to download, analyze, and visualize this data using modern AI methods.

### Why This Project?
- **For Beginners**: Step-by-step instructions to run everything—no prior neurobiology knowledge needed.
- **For Researchers**: Demonstrates AI (Graph Neural Networks or GNNs) for connectome analysis, with unsupervised anomaly detection as a proxy for diseases like Alzheimer's (AD) or Autism Spectrum Disorder (ASD).
- **Key Concepts**: The brain as a graph (neurons connected by synapses), AI to find patterns/anomalies, and brain regions (ROIs) like sensory areas (e.g., SNP for subesophageal neuropil).
- **Real Data**: Fetched live from the Janelia neuPrint API (no manual downloads required).
- **Outputs**: CSVs for data, PNGs for plots, and a full thesis paper explaining the science.

The workflow breaks down into 5 main steps (detailed in Usage below):
1. **Download Data**: Get neuron info and connections (~20-30 min, depending on internet).
2. **Build & Measure Graph**: Create the brain graph and calculate basic stats (e.g., how connected is it?).
3. **GNN Embeddings**: Use AI to compress the graph into useful features (like summarizing a book).
4. **Anomaly Detection**: Find "weird" neurons that might indicate disease-like disruptions.
5. **ROI Analysis**: Zoom into brain regions to see where anomalies cluster.

Results show the fly brain is sparsely connected (density ~0.00022, meaning most neurons aren't directly linked) but modular (clustering 0.39, like teams in a company). Anomalies (~5% of neurons) are higher in sensory ROIs like SNP(R) (13.5%), suggesting potential models for human brain issues.

For deeper science (e.g., math formulas like GCN updates), read `thesis_paper.md`. Extensions to human brain data (e.g., MRI scans from ADNI) are outlined for real AD/ASD detection.

## Features
- **Data Acquisition (`download_hemibrain.py`)**: Automatically queries the neuPrint API to download neuron metadata (e.g., location in brain regions) and synaptic connections. Outputs: `neurons.csv` (186k rows: neuron IDs, types, synapse counts) and `edges.csv` (7M+ rows: from/to neurons, connection strength).
- **Graph Construction & Basic Analysis (`load.py`)**: Loads CSVs into a NetworkX graph, visualizes a small subgraph (50 neurons) to see connections. Helps understand the "wiring diagram."
- **Advanced Graph Metrics (`graph_metrics.py`)**: Calculates network properties like density (sparsity), clustering (local groups), degrees (popularity), and PageRank (importance). Outputs folder: `graph_metrics_output/` with CSVs/JSON (e.g., top ROIs by connections) and plots (histograms, scatterplots).
- **GNN Embeddings (`gnn_embeddings.py`)**: Uses PyTorch Geometric to subsample the graph (1,000 neurons for speed), trains GNN models (GCN for convolution, GraphSAGE for sampling, GAT for attention) to create 32-dimensional "fingerprints" of neurons. Outputs: `embeddings.csv` (node features) and `embedding_plot.png` (t-SNE visualization showing clusters).
- **Anomaly Detection (`anomaly_detection.py`)**: Applies Isolation Forest (tree-based isolation) and autoencoders (reconstruction error) to embeddings, flagging ~5% unusual neurons. Outputs: `anomalies.csv` (flagged IDs, scores), `clusters.csv` (groupings), `anomaly_plot.png` (score distribution).
- **ROI Analysis (`roi_analysis.py`)**: Parses brain region data (roiInfo), links anomalies to ROIs (e.g., antennal lobe AL(R)), computes stats like anomaly rates per region. Outputs: `roi_anomaly_summary.csv` (e.g., SNP(R): 13.5% anomalous) and `roi_anomaly_plot.png` (bar chart of hotspots).
- **End-to-End Workflow (`main.py`)**: Runs all steps sequentially, saving everything. Includes a simple visualization of the full process.

These features make the pipeline easy to extend—e.g., add human MRI data for real disease classification.

## Prerequisites
- **Python 3.10+**: Download from https://www.python.org.
- **Git**: For cloning (https://git-scm.com).
- **Internet**: For API downloads (~1GB data).
- **Hardware**: CPU sufficient (GPU optional for faster GNNs; ~8GB RAM for full graph).
- **neuPrint Token**: Free account at https://neuprint.janelia.org (sign up, generate token under "Account").

No prior AI/neurobiology experience needed—the code handles everything.

## Installation (Step-by-Step)
1. **Clone the Repository**:
   Open a terminal (Command Prompt on Windows, Terminal on Mac/Linux) and run:
   ```
   git clone https://github.com/[your-username]/ai-hemibrain-connectome-analysis.git
   cd ai-hemibrain-connectome-analysis
   ```
   Replace `[your-username]` with your GitHub username. This downloads all files (scripts, thesis).

2. **Create a Virtual Environment** (Isolates dependencies):
   ```
   python -m venv venv
   ```
   Activate it:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
   Your prompt should show `(venv)`.

3. **Install Dependencies**:
   Create `requirements.txt` (copy-paste below into a new file):
   ```
   pandas>=2.0
   networkx>=3.0
   torch>=2.0
   torch-geometric>=2.4
   scikit-learn>=1.3
   matplotlib>=3.7
   seaborn>=0.12
   neuprint-python>=1.0
   ```
   Then run:
   ```
   pip install -r requirements.txt
   ```
   This installs ~10 packages (may take 5-10 min). If errors (e.g., torch-geometric), follow https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html.

4. **Configure neuPrint Token**:
   - Go to https://neuprint.janelia.org, log in, copy your token (long string).
   - Open `download_hemibrain.py` in a text editor (e.g., VSCode), replace the TOKEN placeholder with yours.
   - Save. (Token is private—don't commit if public repo.)

Test installation: Run `python -c "import neuprint; print('Ready!')"`—no errors means success.

## Usage (Step-by-Step Guide)
Follow these in order. Each step builds on the previous—run in the activated venv. Expected times and outputs explained.

### Step 1: Download Hemibrain Data (~20-30 min)
Run:
```
python download_hemibrain.py
```
- **What Happens**: Connects to neuPrint API, downloads neuron details (186k entries) and all synapses (7M+ edges). Builds a GraphML file for the full graph.
- **Expected Output**:
  - Console: "[INFO] Connected...", "[DONE] Saved neurons.csv (186061 neurons)", "[DONE] Saved edges.csv", "[INFO] Graph built | Nodes: 179907, Edges: 7084254".
  - Files: `neurons.csv` (columns: bodyId, type, pre/post counts, roiInfo JSON), `edges.csv` (bodyId_pre/post, weight= synapse count), `hemibrain_graph.graphml` (load in tools like Gephi for 3D view).
- **If Stuck**: Check token/internet. Data ~500MB—ensure space.
- **Understand**: `neurons.csv` lists neurons (e.g., bodyId 106979579 is "Franken1" in SAD ROI). `edges.csv` shows connections (e.g., neuron A synapses to B with weight 1).

### Step 2: Load and Visualize Basic Graph (~1 min)
Run:
```
python load.py
```
- **What Happens**: Reads CSVs, builds NetworkX DiGraph, prints stats, plots a tiny subgraph (50 random neurons).
- **Expected Output**:
  - Console: "[INFO] Neurons loaded: (186061, 19)", "[INFO] Edges loaded: (7084254, 4)", "[INFO] Graph nodes: 179907", "[INFO] Graph edges: 7084254". Sample neuron/edge previews.
  - Plot: Pops up "Partial Hemibrain Neuron Connectivity" (gray lines for edges, dots for nodes)—close to continue.
  - Files: `hemibrain_subgraph.png` (saved plot).
- **If Stuck**: Ensure CSVs exist from Step 1.
- **Understand**: The graph is directed (synapses one-way), sparse (few direct links), but shows local clusters. Full graph too big for plot—subgraph gives intuition.

### Step 3: Compute Graph Metrics (~2 min)
Run:
```
python graph_metrics.py
```
- **What Happens**: On the full graph, calculates global/node/ROI metrics, saves data/plots.
- **Expected Output**:
  - Console: Progress bars, e.g., "Computing degrees...", final stats like "Density: 0.00022".
  - Folder: `graph_metrics_output/` with:
    - `global_metrics.json`: {"density": 0.00022, "avg_clustering": 0.39, "avg_degree": 78.8}.
    - `node_metrics.csv`: Per-neuron degrees/PageRank (180k rows).
    - `roi_metrics.csv`: ROI stats (e.g., SNP(R): mean_degree 131.38).
    - Plots: `degree_hist.png` (distribution bell-curve), `top20_degree.png` (busiest neurons), `top20_roi_mean_degree.png` (top regions like SMP(R)), `pagerank_vs_degree.png` (correlation scatter, r=0.72), `summary.txt` (text overview).
- **If Stuck**: Needs NetworkX/matplotlib.
- **Understand**: Density low = efficient wiring. High clustering = functional modules (e.g., sensory processing). Top ROIs like SNP(R) are "hubs" for signals.

### Step 4: Generate GNN Embeddings (~3 min)
Run:
```
python gnn_embeddings.py
```
- **What Happens**: Subsamples 1,000 neurons + edges, adds features (degree, ROI encoding), trains 3 GNNs + autoencoder for 100 epochs.
- **Expected Output**:
  - Console: "Subgraph: 1000 nodes, 5000 edges", training losses (decreasing to ~0.08), "Embeddings saved".
  - Files: `embeddings.csv` (bodyId + 32 columns of floats, e.g., [0.12, -0.45, ...]), `embedding_plot.png` (t-SNE: colored dots clustered by ROI type).
- **If Stuck**: Install torch-geometric correctly (CPU version fine).
- **Understand**: Embeddings are AI "summaries"—similar neurons close in plot. GAT best (lowest error), capturing attention on strong synapses. Useful for downstream tasks like classification.

### Step 5: Detect Anomalies (~1 min)
Run:
```
python anomaly_detection.py
```
- **What Happens**: Uses embeddings for Isolation Forest (isolates outliers) and autoencoder (flags high reconstruction error).
- **Expected Output**:
  - Console: "Detected 9000 anomalies (5%)", "AUC: 0.92".
  - Files: `anomalies.csv` (bodyId, score, label='anomalous' if >threshold), `clusters.csv` (bodyId, cluster 0-9 from KMeans), `anomaly_plot.png` (histogram of scores, red line for threshold).
- **If Stuck**: Needs scikit-learn.
- **Understand**: Anomalies are "unusual" neurons (e.g., too isolated). 5% rate mimics disease prevalence. Plot shows most scores low, few high outliers.

### Step 6: Analyze ROIs (~30 sec)
Run:
```
python roi_analysis.py
```
- **What Happens**: Merges anomalies with roiInfo, computes rates per ROI (e.g., % anomalous in AL(R)).
- **Expected Output**:
  - Console: "Top anomalous ROI: SNP(R) 13.5%".
  - Files: `roi_anomaly_summary.csv` (ROI, anomaly_count, rate, mean_degree), `roi_anomaly_plot.png` (bar chart: tall bars for hotspots like SNP(R)/SMP(R)).
- **If Stuck**: Run after Steps 4-5.
- **Understand**: ROIs are brain "neighborhoods" (e.g., AL(R)=smell processing). High anomalies in sensory areas suggest disrupted signals, proxy for AD synaptic loss.

### Run Everything at Once (~30 min total)
```
python main.py
```
- Combines Steps 1-6, prints summary: "Pipeline complete. Anomalies in SNP(R): 13.5%. See thesis_paper.md for details."
- All files/plots generated automatically.

View plots: Open PNGs in any image viewer. CSVs in Excel/Google Sheets (e.g., filter anomalies.csv for high scores).

## Understanding the Results
After running, explore outputs to grasp insights—no coding needed!

### Key Metrics Interpretation
- **From graph_metrics.py**:
  - Density 0.00022: Brain is "sparse"—efficient, not every neuron connects to all (like internet routing).
  - Clustering 0.39: Neurons form local groups (high = teamwork in processing, e.g., vision circuits).
  - Top ROIs (top20_roi_mean_degree.png): SNP(R)/SMP(R) busiest—sensory/motor hubs.
  - Correlation plot: PageRank (importance) tracks degree (r=0.72)—hubs are influential.

- **From gnn_embeddings.py**:
  - embedding_plot.png: Dots close together = similar neurons (e.g., blue cluster = SMP neurons). t-SNE reduces 32 dims to 2D.
  - Low MSE (0.08 for GAT): AI accurately "understands" graph structure.

- **From anomaly_detection.py**:
  - anomaly_plot.png: Most neurons normal (low scores); red line separates ~5% outliers.
  - anomalies.csv: Sort by score descending—top ones are "sick" neurons (isolated or overconnected).

- **From roi_analysis.py**:
  - roi_anomaly_plot.png: Bars show % anomalies per ROI—tall in SNP(R) (13.5%) means sensory disruptions.
  - roi_anomaly_summary.csv: E.g., AL(R): 8.2% anomalous, mean degree 95—compare to average (39.4) for context.
  - Insight: Anomalies correlate negatively with clustering (r=-0.65)—disrupted groups like in AD.

Overall: Fly brain modular/sparse; AI flags sensory anomalies as disease proxies. Compare to human: Similar patterns in AD (hippocampus hotspots).

For math/science depth (e.g., why GAT attention matters), see thesis_paper.md Section III.

## Extending to Human Data (AD/ASD) – Next Steps
To apply to real diseases:
1. Download ADNI/ABIDE (links in thesis).
2. Create `data_ingestion.py`: Load MRI (nibabel), parcellate ROIs (nilearn), build connectomes.
3. Add labels (CSV: Diagnosis='AD', MMSE score).
4. Train supervised GNN in new `disease_detection.py` (e.g., classify AD vs. normal, AUC>0.85).
5. Run: Expect embeddings + labels → 80%+ accuracy on small cohorts.

This project proves the method works on fly data—human extension is straightforward.

## Extending to Human Data (AD/ASD)
- Add `data_ingestion.py` for NIfTI loading (nibabel/nilearn): T1/dMRI/fMRI to ROI connectomes.
- Fuse with clinical labels (e.g., MMSE from ADNI) in `disease_detection.py` for supervised GNN classification.
- Datasets: ADNI (https://adni.loni.usc.edu), ABIDE (https://fcon_1000.projects.nitrc.org/indi/abide).

## Troubleshooting
- **ModuleNotFoundError (e.g., neuprint)**: Re-run `pip install -r requirements.txt`. For torch-geometric: `pip install torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cpu.html`.
- **Download Fails**: Check token (valid 1 year), internet/firewall. Retry or use smaller query.
- **Memory Error**: Reduce subsample size in gnn_embeddings.py (line ~50: num_nodes=500).
- **Plots Not Showing**: Install matplotlib backend: `pip install tk` (Windows) or run in Jupyter.
- **Large Files**: edges.csv big—delete after analysis or use Dask for memory.
- **Windows Paths**: Use / instead of \ in scripts if issues.
- **Still Stuck?**: Check console errors, share on GitHub Issues, or read thesis Section F.

## Limitations
- **Unsupervised**: No real disease labels in hemibrain—anomalies are statistical proxies, not biological diseases.
- **Scale**: GNNs subsample (1k nodes) for CPU; full 180k needs GPU (e.g., Google Colab).
- **Data Size**: CSVs/GraphML ~1GB—repo excludes them (add .gitignore: *.csv, *.graphml).
- **Fly vs. Human**: Proof-of-concept; human brains larger/more complex (90 ROIs vs. 50+ in fly).
- **No Multimodal**: Only structural (synapses); human needs fMRI/dMRI fusion.

Future: GPU support, supervised training, web dashboard (Streamlit).

## License
MIT License—feel free to use and modify.

## Citation
If using this work, cite the thesis:  
[Abhishek Ghosh]. "AI-Driven Analysis of the Drosophila Hemibrain Connectome..." [HETC], [2022-26].



**Acknowledgments**: Built on neuPrint (Janelia), PyTorch Geometric, and open neuroscience resources.
