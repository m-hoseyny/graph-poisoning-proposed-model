import torch

def process(batch, gnn_model, edge_classifier_model, device, optimizer, criterion):
    num_nodes_all = [graph.x.shape[0] for graph in batch]
    node_features_all = [graph.x for graph in batch]
    node_feature_batch = torch.cat(node_features_all, dim=0).to(device)
    
    # Get edge indices (already in COO format: [2, num_edges])
    edge_index_all = []
    for graph in batch:
        edge_index_all.append(graph.edge_index)
    
    edge_labels_all = []
    for graph in batch:
        # edge_labels_all.append(graph.edge_attr)
        edge_labels_all.append(torch.argmax(graph.edge_attr, dim=1))
    edge_labels_batch = torch.cat(edge_labels_all, dim=0).to(device)

    # Update the edge indices to reflect the new node ordering in the batched tensor
    # We do this by adding the cumulative sum of the number of nodes from previous graphs
    edge_index_batch = []
    num_nodes_offset = 0
    for i, edge_index in enumerate(edge_index_all):
        edge_index_offset = edge_index + num_nodes_offset
        edge_index_batch.append(edge_index_offset)
        num_nodes_offset += num_nodes_all[i]

    # Concatenate all edge indices along the edge dimension (dim=1)
    edge_index_batch = torch.cat(edge_index_batch, dim=1).to(device)

    # Forward pass through GCN to get node embeddings
    node_embeddings_out = gnn_model(node_feature_batch, edge_index_batch)

    # Prepare edge features by concatenating the embeddings of the head and tail nodes
    edge_embeddings = torch.cat(
        [
            node_embeddings_out[edge_index_batch[0]],
            node_embeddings_out[edge_index_batch[1]],
        ],
        dim=1,
    )

    # Forward pass through MLP for edge classification
    out = edge_classifier_model(edge_embeddings)

    # Compute loss
    loss = criterion(out, edge_labels_batch)

    # Only perform backpropagation if optimizer is provided (training mode)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Accumulate metrics
    total_loss = loss.item()
    preds = out.argmax(dim=1)


    return gnn_model, edge_classifier_model, total_loss, preds, edge_labels_batch


def eval_process(batch, gnn_model, edge_classifier_model, device, optimizer, criterion):
    with torch.no_grad():
        num_nodes_all = [graph.x.shape[0] for graph in batch]
        node_features_all = [graph.x for graph in batch]
        node_feature_batch = torch.cat(node_features_all, dim=0).to(device)
        # Get edge indices (already in COO format: [2, num_edges])
        edge_index_all = []
        for graph in batch:
            edge_index_all.append(graph.edge_index)
        
        edge_labels_all = []
        for graph in batch:
            # edge_labels_all.append(graph.edge_attr)
            edge_labels_all.append(torch.argmax(graph.edge_attr, dim=1))
        edge_labels_batch = torch.cat(edge_labels_all, dim=0).to(device)

        # Update the edge indices to reflect the new node ordering in the batched tensor
        # We do this by adding the cumulative sum of the number of nodes from previous graphs
        edge_index_batch = []
        num_nodes_offset = 0
        for i, edge_index in enumerate(edge_index_all):
            edge_index_offset = edge_index + num_nodes_offset
            edge_index_batch.append(edge_index_offset)
            num_nodes_offset += num_nodes_all[i]

        # Concatenate all edge indices along the edge dimension (dim=1)
        edge_index_batch = torch.cat(edge_index_batch, dim=1).to(device)

        # Forward pass through GCN to get node embeddings
        node_embeddings_out = gnn_model(node_feature_batch, edge_index_batch)

        # Prepare edge features by concatenating the embeddings of the head and tail nodes
        edge_embeddings = torch.cat(
            [
                node_embeddings_out[edge_index_batch[0]],
                node_embeddings_out[edge_index_batch[1]],
            ],
            dim=1,
        )

        # Forward pass through MLP for edge classification
        out = edge_classifier_model(edge_embeddings)

        # Compute loss
        loss = criterion(out, edge_labels_batch)

        # Accumulate metrics
        total_loss = loss.item()
        preds = out.argmax(dim=1)
        
        return gnn_model, edge_classifier_model, total_loss, preds, edge_labels_batch