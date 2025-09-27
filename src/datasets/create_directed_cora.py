import torch
import numpy as np
import os
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from datasets.load_cora import load_directed_cora
import tqdm

def create_directed_cora_compatible_with_planetoid():
    """
    Creates a directed Cora dataset that is compatible with Planetoid node ordering.
    Maps directed Cora node IDs to Planetoid node IDs based on feature vector similarity.
    """
    
    # Load Planetoid Cora dataset
    base_path = '../data'
    planetoid_data = Planetoid(base_path, 'Cora', transform=T.NormalizeFeatures())
    
    # Load directed Cora dataset
    directed_cora = load_directed_cora()
    
    print(f"Planetoid Cora: {planetoid_data.x.shape[0]} nodes, {planetoid_data.edge_index.shape[1]} edges")
    print(f"Directed Cora: {directed_cora.x.shape[0]} nodes, {directed_cora.edge_index.shape[1]} edges")
    
    # Step 1: Create mapping based on exact one-hot feature vector matching
    # Convert to numpy for easier computation
    planetoid_features = planetoid_data.x.numpy()
    directed_features = directed_cora.x.numpy()
    
    print(f"Feature dimensions - Planetoid: {planetoid_features.shape[1]}, Directed: {directed_features.shape[1]}")
    
    # Create mapping: directed_node_id -> planetoid_node_id
    directed_to_planetoid_mapping = {}
    planetoid_to_directed_mapping = {}
    
    # Find exact matches between one-hot feature vectors
    matched_count = 0
    unmatched_directed = []
    
    for directed_node in tqdm.tqdm(range(directed_cora.x.shape[0])):
        directed_features_vec = directed_features[directed_node]
        
        # Find exact match in planetoid features
        found_match = False
        for planetoid_node in range(planetoid_data.x.shape[0]):
            planetoid_features_vec = planetoid_features[planetoid_node]
            
            # Check if vectors are exactly equal (for one-hot vectors)
            if np.array_equal(directed_features_vec, planetoid_features_vec):
                # Make sure this planetoid node hasn't been used yet
                if planetoid_node not in planetoid_to_directed_mapping:
                    directed_to_planetoid_mapping[directed_node] = planetoid_node
                    planetoid_to_directed_mapping[planetoid_node] = directed_node
                    matched_count += 1
                    found_match = True
                    break
        
        if not found_match:
            unmatched_directed.append(directed_node)
    
    print(f"Successfully mapped {matched_count} nodes with exact feature matches")
    print(f"Unmatched directed nodes: {len(unmatched_directed)}")
    
    if len(unmatched_directed) > 0:
        print(f"Warning: {len(unmatched_directed)} directed nodes could not be matched exactly")
        print(f"First few unmatched nodes: {unmatched_directed[:5]}")
    
    # Step 2: Create new node features using planetoid ordering
    # Initialize with zeros
    new_x = torch.zeros((planetoid_data.x.shape[0], directed_cora.x.shape[1]))
    new_y = torch.zeros(planetoid_data.x.shape[0], dtype=torch.long)
    
    # Fill in the mapped features
    for directed_node, planetoid_node in directed_to_planetoid_mapping.items():
        new_x[planetoid_node] = directed_cora.x[directed_node]
        new_y[planetoid_node] = directed_cora.y[directed_node]
    
    # Step 3: Remap edge indices
    new_edge_list = []
    
    for i in range(directed_cora.edge_index.shape[1]):
        source_directed = directed_cora.edge_index[0, i].item()
        target_directed = directed_cora.edge_index[1, i].item()
        
        # Map to planetoid node IDs
        if source_directed in directed_to_planetoid_mapping and target_directed in directed_to_planetoid_mapping:
            source_planetoid = directed_to_planetoid_mapping[source_directed]
            target_planetoid = directed_to_planetoid_mapping[target_directed]
            new_edge_list.append([source_planetoid, target_planetoid])
    
    # Convert to tensor
    if new_edge_list:
        new_edge_index = torch.tensor(new_edge_list, dtype=torch.long).t()
    else:
        new_edge_index = torch.empty((2, 0), dtype=torch.long)
    
    print(f"Remapped edges: {new_edge_index.shape[1]} edges")
    
    # Step 4: Create final directed graph
    directed_cora_planetoid_compatible = Data(
        x=new_x,
        edge_index=new_edge_index,
        y=new_y
    )
    
    # Verification
    print(f"\nFinal dataset:")
    print(f"Nodes: {directed_cora_planetoid_compatible.x.shape[0]}")
    print(f"Features: {directed_cora_planetoid_compatible.x.shape[1]}")
    print(f"Edges: {directed_cora_planetoid_compatible.edge_index.shape[1]}")
    print(f"Classes: {directed_cora_planetoid_compatible.y.max().item() + 1}")
    
    return directed_cora_planetoid_compatible, directed_to_planetoid_mapping

class DirectedCoraDataset(InMemoryDataset):
    def __init__(self, root, data_obj=None, transform=None, pre_transform=None):
        self.data_obj = data_obj
        super().__init__(root, transform, pre_transform)
        if os.path.exists(self.processed_paths[0]):
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        if self.data_obj is not None:
            data_list = [self.data_obj]
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])

def save_directed_cora_dataset(data, mapping, save_root='/mnt/data/mohammad-hosseini/DiGress/SaGess/data/DirectedCora'):
    """
    Save the directed Cora dataset as an InMemoryDataset that can be loaded again.
    """
    # Create the dataset directory if it doesn't exist
    os.makedirs(save_root, exist_ok=True)
    
    # Save the node mapping separately
    mapping_path = os.path.join(save_root, 'node_mapping.pt')
    torch.save(mapping, mapping_path)
    
    # Create and save the InMemoryDataset
    dataset = DirectedCoraDataset(save_root, data)
    
    print(f"DirectedCora InMemoryDataset saved to: {save_root}")
    print(f"Node mapping saved to: {mapping_path}")
    print(f"To load: dataset = DirectedCoraDataset('{save_root}', None)")
    
    return save_root


if __name__ == "__main__":
    # Execute the mapping
    directed_cora_compatible, node_mapping = create_directed_cora_compatible_with_planetoid()

    # Save the dataset
    save_path = save_directed_cora_dataset(directed_cora_compatible, node_mapping)
    print(f"Directed Cora dataset compatible with Planetoid saved successfully!")
