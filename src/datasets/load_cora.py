import os
import requests
import torch
import numpy as np
from torch_geometric.data import Data

def download_cora_raw(path='./dataset/data/cora_raw'):
    """Downloads the raw Cora dataset files if they don't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        
    cites_url = 'https://linqs-data.soe.ucsc.edu/public/lbc/cora.cites'
    content_url = 'https://linqs-data.soe.ucsc.edu/public/lbc/cora.content'
    
    cites_path = os.path.join(path, 'cora.cites')
    content_path = os.path.join(path, 'cora.content')

    if not os.path.exists(cites_path):
        print("Downloading cora.cites...")
        response = requests.get(cites_url)
        with open(cites_path, 'wb') as f:
            f.write(response.content)
            
    if not os.path.exists(content_path):
        print("Downloading cora.content...")
        response = requests.get(content_url)
        with open(content_path, 'wb') as f:
            f.write(response.content)
            
    return cites_path, content_path

def load_directed_cora():
    """Loads the original directed Cora dataset into a PyG Data object."""
    cites_path, content_path = download_cora_raw()

    # 1. Load content file to get node features, labels, and ID mapping
    print("Processing cora.content...")
    with open(content_path, 'r') as f:
        content = f.read().strip().split('\n')
        
    # Map paper IDs (strings) to integer indices (0 to N-1)
    paper_ids = [line.split('\t')[0] for line in content]
    id_to_idx = {paper_id: i for i, paper_id in enumerate(paper_ids)}
    
    # Parse features (binary bag-of-words) and labels
    features = np.array([line.split('\t')[1:-1] for line in content], dtype=np.float32)
    labels_str = [line.split('\t')[-1] for line in content]
    
    # Map string labels to integer indices
    unique_labels = sorted(list(set(labels_str)))
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    labels = [label_to_idx[label] for label in labels_str]

    # 2. Load cites file to get directed edges
    print("Processing cora.cites...")
    with open(cites_path, 'r') as f:
        cites = f.read().strip().split('\n')

    source_nodes = []
    target_nodes = []
    
    # Each line is "cited_paper_id citing_paper_id"
    # This corresponds to an edge: cited -> citing
    for edge in cites:
        parts = edge.split('\t')
        cited_id, citing_id = parts[0], parts[1]
        
        # Ensure both papers are in our content mapping
        if cited_id in id_to_idx and citing_id in id_to_idx:
            source_nodes.append(id_to_idx[cited_id])
            target_nodes.append(id_to_idx[citing_id])
            
    # 3. Create PyG Data object
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    
    # Create a directed graph data object
    directed_cora_data = Data(x=x, edge_index=edge_index, y=y)
    
    return directed_cora_data

# # Load the data and print its properties
# cora_directed_graph = load_directed_cora()

# print("\n--- Directed Cora Graph Properties ---")
# print(f"Graph object: {cora_directed_graph}")
# print(f"Number of nodes: {cora_directed_graph.num_nodes}")
# print(f"Number of edges: {cora_directed_graph.num_edges}") # This should be 5,429
# print(f"Is the graph directed? {'Yes' if cora_directed_graph.is_directed() else 'No'}")