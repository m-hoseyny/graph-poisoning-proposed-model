import os
import random
import time 
import networkx as nx

random.seed(120)
import copy
import torch
torch.manual_seed(120)
from torch.utils.data import random_split, Dataset
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_undirected, k_hop_subgraph
from torch_geometric.datasets import EmailEUCore, Planetoid, AttributedGraphDataset, SNAPDataset, WikipediaNetwork
from datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
import torch.nn.functional as F
import torch_geometric.transforms as T
import pickle
from tqdm import tqdm
import multiprocessing as mp 
from utils import random_walk, rw_task, ego_task

from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import degree

from typing import List

class CoraDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg 
        self.data_file = self.cfg.dataset.name
        self.per_node_samples_rw = self.cfg.dataset.per_node_samples_rw
        self.per_node_samples_unif = self.cfg.dataset.per_node_samples_unif
        self.per_node_samples_ego = self.cfg.dataset.per_node_samples_ego
        self.subgraph_size = self.cfg.dataset.subgraph_size
        self.ego_sample_radius = self.cfg.dataset.ego_sample_radius
        self.mask_number = self.cfg.dataset.mask_number
        self.original_node_attribute = self.cfg.dataset.original_node_attribute
        self.edge_att_dict = {}
        self.removed_samples: List[Data] = []
        self.test_dataset: List[Data] = []
        self.test_dataset_original_edges = []

        self.attribute_task = self.cfg.general.attribute_task
        
        # Use absolute path based on project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        print(f'-------\ntask {self.attribute_task}\n-------')
        # dataset_path = os.path.join(project_root, 'datasets', 'directed_cora_attributes.pt')
        if self.attribute_task == 'class':
            dataset_path = os.path.join(project_root, 'datasets', f'new_EPAGCL_{self.data_file}_{self.cfg.gnn_model.name}_class_performance.pkl')
        elif self.attribute_task == 'edge':
            dataset_path = os.path.join(project_root, 'datasets', f'EPAGCL_Full_{self.data_file}_{self.cfg.gnn_model.name}_edge_performance.pkl')
        print('Edge performaces: ', dataset_path)
        # if cfg.gnn_model.name == 'gcn' and self.data_file == 'Cora':
        #     dataset_path = os.path.join(project_root, 'datasets', 'EPAGCL_Full_Cora_gcn_edge_performance.pkl')
        # # elif cfg.gnn_model.name == 'sage' and self.data_file == 'Cora':
        # #     dataset_path = os.path.join(project_root, 'datasets', 'Cora_sage_edge_performance.pkl')
        # elif cfg.gnn_model.name == 'gat' and self.data_file == 'Cora':
        #     dataset_path = os.path.join(project_root, 'datasets', 'EPAGCL_Full_Cora_gat_edge_performance.pkl')
        
        # if cfg.gnn_model.name == 'gcn' and self.data_file == 'CiteSeer':
        #     dataset_path = os.path.join(project_root, 'datasets', 'EPAGCL_Full_CiteSeer_gcn_edge_performance.pkl')
        # # elif cfg.gnn_model.name == 'sage' and self.data_file == 'CiteSeer':
        # #     dataset_path = os.path.join(project_root, 'datasets', 'new_CiteSeer_sage_edge_performance.pkl')
        # elif cfg.gnn_model.name == 'gat' and self.data_file == 'CiteSeer':
        #     dataset_path = os.path.join(project_root, 'datasets', 'EPAGCL_Full_CiteSeer_gat_edge_performance.pkl')
            
        # if cfg.gnn_model.name == 'gcn' and self.data_file == 'Wikipedia':
        #     dataset_path = os.path.join(project_root, 'datasets', 'new_Wikipedia_gcn_edge_performance.pkl')
        # elif cfg.gnn_model.name == 'sage' and self.data_file == 'Wikipedia':
        #     dataset_path = os.path.join(project_root, 'datasets', 'new_Wikipedia_sage_edge_performance.pkl')
        # elif cfg.gnn_model.name == 'gat' and self.data_file == 'Wikipedia':
        #     dataset_path = os.path.join(project_root, 'datasets', 'new_Wikipedia_gat_edge_performance.pkl')
        
        with open(dataset_path, 'rb') as f:
            self.edge_gcn_attributes = pickle.load(f)
        
        base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, 'data')
        directed_graph = False
        
        if self.data_file == 'Wikipedia':
            graphs = WikipediaNetwork(base_path, name='chameleon', transform=T.NormalizeFeatures())
            directed_graph = False
        else:
            graphs = Planetoid(base_path, self.data_file, transform=T.NormalizeFeatures())
        # graphs.data = load_directed_cora()
        # graphs = DirectedCoraDataset('/mnt/data/mohammad-hosseini/DiGress/SaGess/data/DirectedCora')
            directed_graph = True 

        self.graphs = graphs
        # print(graphs.data.has_isolated_nodes())
        
        if directed_graph == True:
            edge_list = graphs[0].edge_index
            edge_list_nx = [(int(i[0]), int(i[1])) for i in edge_list.transpose(0, 1)]
            self.nx_graph = nx.from_edgelist(edge_list_nx, create_using=nx.DiGraph)
        else:
            edge_list = to_undirected(graphs[0].edge_index)
            edge_list_nx = [(int(i[0]), int(i[1])) for i in edge_list.transpose(0, 1)]
            self.nx_graph = nx.from_edgelist(edge_list_nx)


        # if self.mask_number and self.mask_number > 0:
        #     print(f"Masking {self.mask_number} nodes")
        #     self.mask_nodes(self.mask_number)
        #     print('edges in graph : ', len(list(self.nx_graph.edges())))
        
        #nx_graph = nx.k_core(nx_graph, k= 27)
        print('edges in graph : ', len(list(self.nx_graph.edges())))

        y = torch.zeros([1, 0]).float()
        n_nodes = len(list(self.nx_graph.nodes()))
        print('nodes in graph : ', n_nodes)

        self.samples_sizes = []
        self.dataset_node_lists = []

        sampling_start_time = time.time()

        if self.cfg.dataset.sampling_method == 'mix':
            print('Sampling type: mix')

            self.random_walk_sample(per_node_samples=30, 
                                    subgraph_size=self.subgraph_size)
            self.uniform_sample(per_node_samples=7000, 
                                subgraph_size=self.subgraph_size)
            # self.ego_sample(per_node_samples=7, 
                            #   subgraph_size=self.subgraph_size)
            self.ego_sample_threaded(per_node_samples=7, 
                                        subgraph_size=self.subgraph_size, 
                                        radius=self.ego_sample_radius,
                                        max_workers=self.cfg.dataset.sampling_threads)

        elif self.cfg.dataset.sampling_method == 'random_walk':
            print('Sampling type: random walk')

            self.random_walk_sample(self.per_node_samples_rw, self.subgraph_size)
            
            # self.random_walk_sample_threaded(self.per_node_samples_rw, self.subgraph_size, 
            #                                  max_workers=self.cfg.dataset.sampling_threads)

        elif self.cfg.dataset.sampling_method == 'ego':
            print('Sampling type: ego')

            # self.ego_sample(self.per_node_samples_ego, self.subgraph_size, 
                                                        # radius=self.ego_sample_radius)

            self.ego_sample_threaded(self.per_node_samples_ego, self.subgraph_size, 
                                        radius=self.ego_sample_radius, 
                                        max_workers=self.cfg.dataset.sampling_threads)

        elif self.cfg.dataset.sampling_method == 'uniform':
            print('Sampling type: uniform')

            self.uniform_sample(self.per_node_samples_unif, self.subgraph_size)
        
        sampling_end_time = time.time()
        n_samples = len(self.dataset_node_lists)
        print(f'We need to sample {n_samples} subgraphs')
        print(f'We sampled {len(self.dataset_node_lists)} subgraphs')
        self.sample_size = len(self.dataset_node_lists)
        sampling_time = sampling_end_time - sampling_start_time
        minutes = sampling_time // 60
        seconds = sampling_time % 60
        print('sampling start time : ', sampling_start_time)
        print('sampling end time : ', sampling_end_time)
        print(f"Total sampling time: {int(minutes)} mins, {seconds:.2f} secs")
        
        # sampled graphs effectively for training: 90%
        dataset_samples_initialids = [(self.dataset_node_lists[i],
                                        subgraph(torch.tensor(self.dataset_node_lists[i]), edge_list)[0])
                                        for i in range(n_samples)]

        dict_maps = [{dataset_samples_initialids[j][0][i]: i for i in range(self.samples_sizes[j])} for j in range(n_samples)]
        dataset_samples_wnmaps = [(torch.tensor([[x] for x in dataset_samples_initialids[i][0]]),
                                    dataset_samples_initialids[i][1].apply_(lambda x: dict_maps[i][x])) for i in
                                    range(n_samples)]

        Train_data = []
        self.dataset_samples_wnmaps = dataset_samples_wnmaps
        undirected_transformer = ToUndirected()
        skiped = 0
        for i in tqdm(range(n_samples), desc='Creating attributes'):
            # edge_attr = torch.tensor(
            #     [[0 for k in range(dataset_samples_wnmaps[i][1].size()[1])],
            #      [1 for n in range(dataset_samples_wnmaps[i][1].size()[1])]], 
            #     dtype = torch.long).transpose(0,1)
        
            edge_attr = self.get_edge_attributes(dataset_samples_wnmaps[i][0], 
                                                    dataset_samples_wnmaps[i][1])
            if self.original_node_attribute:
                x = self.graphs[0].x[dataset_samples_wnmaps[i][0].flatten()]
            else:
                x = F.one_hot(torch.flatten(dataset_samples_wnmaps[i][0]), 
                                num_classes = n_nodes).float().to_sparse()
            local_data = Data(x=x,
                            edge_index=dataset_samples_wnmaps[i][1],
                            edge_attr = edge_attr,
                            n_nodes=self.subgraph_size*torch.ones(1, dtype=torch.long), y = y)
            Train_data.append(local_data)
        
        self.original_data = copy.deepcopy(Train_data)
        with open('edge_attributes_dict.pkl', 'wb') as f:
            pickle.dump(self.edge_att_dict, f)
        # Remove samples connected to randomly selected nodes if requested
        if self.cfg.dataset.get('remove_inference_nodes', False):
            num_nodes_to_remove = self.cfg.dataset.get('num_inference_nodes', 100)
            self.remove_nodes_for_inference(Train_data, num_nodes_to_remove)
            self.create_test_dataset()
        self.data = Train_data

    
    def get_edge_attributes(self, x, edge_index, test=False):
        edge_att = []
        edge_pairs = []  # store undirected pairs for reconstruction
        center_node_id = x[0].item()
        edge_size = edge_index.size()
        classes = self.cfg.dataset.edge_attribute_classes
        node_mapper = {}
        # Step 1: collect edge attributes in original order
        for j in range(edge_size[1]):
            node_1 = edge_index[0, j].item()
            node_2 = edge_index[1, j].item()
            if not test:
                node_1_id = x[node_1].item()
                node_2_id = x[node_2].item()
            else:
                node_1_id = node_1
                node_2_id = node_2
            # node_1_id, node_2_id = sorted([node_1_id, node_2_id])
            if (node_1_id, node_2_id) in self.edge_gcn_attributes[node_1_id]:
                local_edge_att = self.edge_gcn_attributes[node_1_id][(node_1_id, node_2_id)][-1][0]
            elif (node_2_id, node_1_id) in self.edge_gcn_attributes[node_1_id]:
                local_edge_att = self.edge_gcn_attributes[node_1_id][(node_2_id, node_1_id)][-1][0]
            
            edge_att.append(local_edge_att)
            node_mapper[(node_1, node_2)] = (node_1_id, node_2_id)
            edge_pairs.append((node_1, node_2))  # keep directed for reconstruction
            

        # edge_att = torch.tensor(edge_att)
        unique_edges = dict(zip(edge_pairs, edge_att))

        # Step 3: rank unique edges by attribute values
        sorted_edges = sorted(unique_edges.items(), key=lambda kv: kv[1], reverse=True)
        edge_labels = {}
        batches = len(sorted_edges) // classes if len(sorted_edges) >= classes else len(sorted_edges)

        for k, (uv, val) in enumerate(sorted_edges):
            for th in range(classes ):
                if th*batches <= k < (th+1)*batches:
                    label = th 
                    break
            edge_labels[uv] = label
            self.edge_att_dict[node_mapper[uv]] = label
        
        labels = []
        for u, v in edge_pairs:
            uv = (u, v)
            labels.append(edge_labels[uv])
            # print(uv, edge_labels[uv])

        # Step 4: reconstruct labels in original order (symmetry guaranteed)
        labels = list(labels)
        labels = torch.tensor(labels, dtype=torch.int8)
        labels = F.one_hot(labels.long(), num_classes=classes).float()

        edge_attr = labels
        return edge_attr

    def uniform_sample(self, per_node_samples=5000, subgraph_size=20):
        n_nodes = len(list(self.nx_graph.nodes()))
        self.samples_sizes += [subgraph_size for i in range(per_node_samples)]
        self.dataset_node_lists+=[list(random.choices(range(n_nodes), k=subgraph_size)) for i in tqdm(range(per_node_samples), desc="Uniform Sampling")]


    def random_walk_sample(self, per_node_samples=30, subgraph_size=20):
        G_copy = copy.copy(self.nx_graph)
        G_copy = G_copy.to_undirected()
        self.dataset_node_lists = []
        
        for n in tqdm(self.nx_graph.nodes(), desc="Random Walk Sampling"):
            for _ in range(per_node_samples):
                final_sample = random_walk(G_copy, n, subgraph_size)
                self.samples_sizes.append(subgraph_size)
                self.dataset_node_lists.append(final_sample)


    def ego_sample(self, per_node_samples=10, subgraph_size=20, radius=2):
        for n in tqdm(self.nx_graph.nodes(), desc="Ego Sampling"):
            ego_net = nx.ego_graph(self.nx_graph.to_undirected(), n, radius=radius)

            temp_initial_size = len(list(ego_net.nodes()))
            for _ in range(per_node_samples):
                temp_ego = copy.deepcopy(ego_net)
                temp_size = temp_initial_size
                final_sample = list(temp_ego.nodes())
                while temp_size > subgraph_size:
                    n_nodes_to_burn = int(temp_size / 2)
                    temp_burning = copy.deepcopy(list(temp_ego.nodes()))
                    temp_burning.remove(n)
                    burned_nodes = list(random.choices(temp_burning, k=n_nodes_to_burn))
                    temp_ego.remove_nodes_from(burned_nodes)
                    final_sample = list(set(list(max(nx.connected_components(temp_ego), key=len)) + [n]))
                    temp_size = len(final_sample)

                self.samples_sizes.append(temp_size)

                self.dataset_node_lists.append(final_sample)

    # def mask_nodes(self, mask_number = 100):
    #     '''
    #         This function is going to mask nodes for training Graph Diffusion Model phase
    #         These nodes are going to be removed from the graph, to mask nodes to not be used in the training phase.
    #         After training, the model must be able to predict these nodes.
    #         So, first, deepcopy from main graph. Remove nodes from main graph.
    #         mask_number: number of nodes to be masked (removed)
    #     '''
    #     self.main_nx_graph = copy.deepcopy(self.nx_graph)
    #     # Fixed seed to remove the same nodes in each run
    #     node_list = list(random.choices(list(self.main_nx_graph.nodes()), k=mask_number))
    #     self.nx_graph.remove_nodes_from(node_list) 
        
    def remove_nodes_for_inference(self, train_data, num_nodes=100):
        """
        Remove samples connected to randomly selected nodes.
        These nodes will be used during inference phase.
        
        Args:
            train_data: List of Data objects representing the training samples
            num_nodes: Number of nodes to remove from training data
        """
        print(f"Removing samples connected to {num_nodes} randomly selected nodes for inference")
        
        # Get all unique nodes from the dataset
        all_nodes = set()
        for sample in self.dataset_node_lists:
            all_nodes.update(sample)
        all_nodes = list(all_nodes)
        
        # Randomly select nodes to remove
        random.seed(42)  # For reproducibility
        edge_index = self.graphs[0].edge_index
        num_nodes = self.graphs[0].num_nodes
        node_degrees = degree(edge_index[0], num_nodes=num_nodes).to('cpu')
        nodes_with_min_edges = (node_degrees >= self.cfg.dataset.min_number_edges).nonzero(as_tuple=True)[0]
        selected_nodes = nodes_with_min_edges
        selected_nodes_set = set(selected_nodes.tolist()) 
        # self.removed_nodes = random.sample(all_nodes, min(num_nodes, len(all_nodes)))
        self.removed_nodes = list(selected_nodes_set)
        with open('./removed_nodes.pkl', 'wb') as f:
            pickle.dump(self.removed_nodes, f)
        print(f"Selected {len(self.removed_nodes)} nodes for inference")
        
        # Find and remove samples that contain any of the removed nodes
        indices_to_keep = []
        indices_removed = []
        for i, sample in enumerate(self.dataset_node_lists):
            # Check if sample contains any of the removed nodes
            if not any(node in self.removed_nodes for node in sample):
                indices_to_keep.append(i)
            else:
                indices_removed.append(i)
        
        # Update the training data
        filtered_data = [train_data[i] for i in indices_to_keep]
        self.removed_samples = [train_data[i] for i in indices_removed]
        removed_count = len(train_data) - len(filtered_data)
        print(f"Removed {removed_count} samples connected to selected nodes")
        
        # Update the class attributes
        self.data = filtered_data
        self.sample_size = len(filtered_data)
        
        return filtered_data
    
    def create_test_dataset(self):
        # radius = self.cfg.dataset.ego_sample_radius
        radius = 1
        for node_id in self.removed_nodes:
            subset, edge_index, mapping, edge_mask = k_hop_subgraph(node_id, 
                                                                    num_hops=radius, 
                                                                    edge_index=self.graphs[0].edge_index,
                                                                    relabel_nodes=True)
            original_edge_index = self.graphs[0].edge_index[:, edge_mask]
            edge_attr = self.get_edge_attributes(subset.unsqueeze(1), original_edge_index, test=True)
            x = self.graphs[0].x[subset]
            ego_graph = Data(x=x, edge_index=edge_index, 
                             edge_attr=edge_attr, 
                             original_edge_index=original_edge_index, 
                             target_node_id=node_id,
                             center_node_mapping=mapping[0].item())
            self.test_dataset.append(ego_graph)
        return self.test_dataset

    # Note: slower than regular rw 
    def random_walk_sample_threaded(self, per_node_samples=30, subgraph_size=20, max_workers=2):
        G = copy.copy(self.nx_graph)

        args_list = [(n, G, per_node_samples, subgraph_size) for n in self.nx_graph.nodes()]

        with mp.Pool(processes=max_workers) as pool:
            results = list(tqdm(pool.imap(rw_task, args_list), total=len(args_list), desc="Threaded Random Walk Sampling"))

        all_samples, all_sizes = zip(*results) 
        # Flatten the list of lists
        all_samples = [item for sublist in all_samples for item in sublist]  
        all_sizes = [item for sublist in all_sizes for item in sublist]
        
        self.dataset_node_lists.extend(all_samples)
        self.samples_sizes.extend(all_sizes)


    def ego_sample_threaded(self, per_node_samples=10, subgraph_size=20, radius=2, max_workers=2):
        G = copy.copy(self.nx_graph)

        args_list = [(n, G, radius, per_node_samples, subgraph_size) for n in self.nx_graph.nodes()]

        with mp.Pool(processes=max_workers) as pool:
            results = list(tqdm(pool.imap(ego_task, args_list), total=len(args_list), desc="Threaded Ego Sampling"))

        all_samples, all_sizes = zip(*results) 
        # Flatten the list of lists
        all_samples = [item for sublist in all_samples for item in sublist]  
        all_sizes = [item for sublist in all_sizes for item in sublist]
        
        self.dataset_node_lists.extend(all_samples)
        self.samples_sizes.extend(all_sizes)


    def __len__(self):
        return self.sample_size

    def __getitem__(self, idx):
        # adj = self.adjs[idx]
        # n = adj.shape[-1]
        # X = torch.ones(n, 1, dtype=torch.float)
        # y = torch.zeros([1, 0]).float()
        # edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
        # edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
        # edge_attr[:, 1] = 1
        # num_nodes = n * torch.ones(1, dtype=torch.long)
        # data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr,
        #                                  y=y, idx=idx, n_nodes=num_nodes)
        graph = self.data[idx]
        return graph


class CoraDataModule(AbstractDataModule):
    def __init__(self, cfg, n_graphs= 100000):
        super().__init__(cfg)
        self.n_graphs = n_graphs/2
        self.graphs = CoraDataset(cfg)
        self.prepare_data()
        self.inner = self.train_dataloader()
        
    def __getitem__(self, item):
        return self.inner[item]

    def prepare_data(self):
        #print(len(self.graphs))
        test_len = int(round(len(self.graphs) * 0.05))
        train_len = int(round((len(self.graphs) - test_len) * 0.95))
        #print(train_len)
        val_len = len(self.graphs) - train_len - test_len
        print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')
        splits = random_split(self.graphs, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(1234))

        datasets = {'train': splits[0], 'val': splits[1], 'test': self.graphs.removed_samples}
        super().prepare_data(datasets)


class CoraDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule):
        self.datamodule = datamodule
        self.name = 'nx_graphs'
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = self.datamodule.node_types()             # There are no node types
        print('Node types: ', self.node_types)
        self.edge_types = self.datamodule.edge_counts()
        print('Edge types: ', self.edge_types)
        super().complete_infos(self.n_nodes, self.node_types)
