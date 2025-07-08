import pandas as pd
import numpy as np
from tqdm import tqdm
import math

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn import Sequential, Linear, ReLU, MultiheadAttention, Dropout, LayerNorm, AvgPool1d
from torchmetrics.functional import mean_absolute_error
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from torch.nn.init import zeros_,xavier_normal_

import os

import networkx as nx
import torch_geometric.data as Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv
from torch_geometric.transforms import NormalizeScale
from torch_geometric.data import Batch
from torchmetrics.functional import mean_absolute_error
from torch.utils.data import random_split

import matplotlib.pyplot as plt

from biopandas.pdb import PandasPdb
import periodictable
from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

import logging, sys


max_antibody_sequence_length = 669
max_antigen_sequence_length = 3102

class ProteinDataset(Dataset):
    def __init__(self, df, max_antibody_sequence_length, max_antigen_sequence_length, save_dir='./content/graph_data'):
        self.sequences = df['Ab'].values
        self.viruses = df['Ag'].values
        self.labels = df['log(IC50)'].values
        self.max_antibody_sequence_length = max_antibody_sequence_length
        self.max_antigen_sequence_length = max_antigen_sequence_length
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        virus = self.viruses[idx]
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.float64)

        antibody = self.load_or_generate_graph(sequence, self.max_antibody_sequence_length, 'antibodies')
        antigen = self.load_or_generate_graph(virus, self.max_antigen_sequence_length, 'antigens')

        if antibody is not None and antigen is not None:
            return antibody, antigen, label
        else:
            return self.__getitem__((idx + 1) % len(self))  # Ensure idx is within bounds

    def load_or_generate_graph(self, pdb_file, max_sequence_length, graph_type):
        graph_path = os.path.join(self.save_dir, f'{graph_type}_{pdb_file}.pt')

        if os.path.exists(graph_path):
            return torch.load(graph_path)

        if graph_type == 'antigens':
            graph_constructed = self.pdb_to_graph_virus(pdb_file, max_sequence_length)
        else:
            graph_constructed = self.pdb_to_graph_antibody(pdb_file, max_sequence_length)

        if graph_constructed is not None:
            torch.save(graph_constructed, graph_path)
        return graph_constructed

    def pdb_to_graph_virus(self, pdb_file, max_antigen_sequence_length):
        return self.pdb_to_graph(pdb_file, max_antigen_sequence_length, 'antigens')

    def pdb_to_graph_antibody(self, pdb_file, max_antibody_sequence_length):
        return self.pdb_to_graph(pdb_file, max_antibody_sequence_length, 'antibodies')

    def pdb_to_graph(self, pdb_file, max_sequence_length, graph_type):
        seq_length = max_sequence_length
        amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
        aa_to_index = {aa: i for i, aa in enumerate(amino_acids)}
        pdbparser = PDBParser()

        try:
            structure = pdbparser.get_structure(pdb_file, os.path.join(f'./content/{graph_type}/'+pdb_file+'.pdb'))
            chains = {chain.id: seq1(''.join(residue.resname for residue in chain)) for chain in structure.get_chains()}
            full_sequence = ''
            for value in chains.values():
                full_sequence += value
            full_sequence = full_sequence.replace("X", "")

            if len(full_sequence) > seq_length:
                print(f"Exceeds max length {graph_type}")
                return None

            indices = [aa_to_index[aa] for aa in full_sequence]
            encoded = F.one_hot(torch.tensor(indices), num_classes=len(amino_acids)).float()
            padded_encoded = F.pad(encoded.flatten(), (0, max(seq_length * len(amino_acids) - encoded.flatten().shape[0], 0)))

            ppdb = PandasPdb()
            ppdb.read_pdb(os.path.join(f'./content/{graph_type}/'+pdb_file+'.pdb'))
            coords = ppdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']].values
            atomic_nums = ppdb.df['ATOM']['element_symbol'].apply(lambda symbol: periodictable.elements.symbol(symbol).number).values

            graph = nx.Graph()
            num_atoms = len(coords)
            for i in range(num_atoms):
                graph.add_node(i, x=coords[i][0], y=coords[i][1], z=coords[i][2], atomic_number=atomic_nums[i])
            for i in range(num_atoms):
                for j in range(i + 1, num_atoms):
                    dist = ((coords[i] - coords[j]) ** 2).sum() ** 0.5
                    if dist < 5:
                        bond_strength = 1 / dist
                        graph.add_edge(i, j, distance=dist, bond_strength=bond_strength)

            edge_attrs = {}

            for u, v, data in graph.edges(data=True):
                edge_attrs[(u, v)] = [data['distance'], data['bond_strength']]
                edge_attrs[(v, u)] = [data['distance'], data['bond_strength']]

            data = Data.Data(
                x=torch.tensor(list(nx.get_node_attributes(graph, 'x').values())).to(torch.float64),
                y_coord=torch.tensor(list(nx.get_node_attributes(graph, 'y').values())).to(torch.float64),
                z_coord=torch.tensor(list(nx.get_node_attributes(graph, 'z').values())).to(torch.float64),
                pos=torch.tensor(coords).to(torch.float64),
                edge_index=torch.tensor(list(graph.edges)).to(torch.float64).t().contiguous(),
                edge_attr=torch.tensor([edge_attrs[e] for e in graph.edges()]).to(torch.float64),
                z=torch.tensor(list(nx.get_node_attributes(graph, 'atomic_number').values())).to(torch.float64),
                seq=padded_encoded.to(torch.float64),
                y = torch.tensor([0.0]).to(torch.float64)
            )
            return data

        except Exception as e:
            print(e)
            return None

antibody_name="10-1074"
antigen_name="0013095_2_11"
test_df = pd.DataFrame({
            "Ab": [antibody_name],
            "Ag": [antigen_name],
            "log(IC50)": 0.0
        })
test_ds = ProteinDataset(test_df, max_antibody_sequence_length, max_antigen_sequence_length)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)



class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=16):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}")
        self.head_dim = embed_dim // num_heads
        self.query_dense = nn.Linear(embed_dim, embed_dim)
        self.key_dense = nn.Linear(embed_dim, embed_dim)
        self.value_dense = nn.Linear(embed_dim, embed_dim)
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def forward(self, inputs):
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = query.view(-1, self.num_heads, self.head_dim)
        key = key.view(-1, self.num_heads, self.head_dim)
        value = value.view(-1, self.num_heads, self.head_dim)
        query = query.permute(1, 0, 2)
        key = key.permute(1, 0, 2)
        value = value.permute(1, 0, 2)
        dot_product = torch.matmul(query, key.permute(0, 2, 1))
        scaled_dot_product = dot_product / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention_weights = torch.softmax(scaled_dot_product, dim=-1)
        output = torch.matmul(attention_weights, value)
        output = output.permute(1, 0, 2)
        output = output.view(-1, self.embed_dim)
        output = self.combine_heads(output)
        return output

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dense_dim=1024, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_dim, num_heads)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dense1 = nn.Linear(embed_dim, dense_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dense2 = nn.Linear(dense_dim, embed_dim)

    def forward(self, inputs):
        attention_output = self.attention(inputs)
        attention_output = self.dropout1(attention_output)
        output1 = self.norm1(inputs + attention_output)
        dense_output = self.dense1(output1)
        dense_output = self.dropout2(dense_output)
        output2 = self.norm2(output1 + dense_output)
        output = self.dense2(output2)
        return output

class CrossAttention(nn.Module):
    def __init__(self, dim, input_shape):
        super(CrossAttention, self).__init__()
        self.dim = dim
        self.input_shape = input_shape

        self.Wq = nn.Parameter(torch.Tensor(input_shape[0][-1], self.dim))
        self.Wk = nn.Parameter(torch.Tensor(input_shape[1][-1], self.dim))
        self.Wv = nn.Parameter(torch.Tensor(input_shape[1][-1], self.dim))

        nn.init.xavier_uniform_(self.Wq)
        nn.init.xavier_uniform_(self.Wk)
        nn.init.xavier_uniform_(self.Wv)

    def forward(self, inputs):
        x, y = inputs

        Q = torch.matmul(x, self.Wq)
        K = torch.matmul(y, self.Wk)
        V = torch.matmul(y, self.Wv)

        attn_weights = torch.matmul(Q, K.t()) / torch.sqrt(torch.tensor(self.dim, dtype=torch.float64))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = attn_weights * V
        output = torch.cat([x, attn_output], dim=-1)

        return output

class CombinedModel(nn.Module):
    def __init__(self, hidden_channels=128, num_layers=16):
        super(CombinedModel, self).__init__()

        self.cross_attn_1 = CrossAttention(128, [(1024,),(1024,)])
        self.cross_attn_2 = CrossAttention(128,[(1024,),(1024,)])
        self.cross_attn = CrossAttention(128,[(1152,),(1152,)])
        self.self_atten_1 = nn.AdaptiveAvgPool1d(1)
        self.self_atten_2 = nn.AdaptiveAvgPool1d(1)
        self.cross_pooling = nn.AdaptiveAvgPool1d(1)
        self.dense = nn.Linear(1280, 256)
        self.output_layer1 = nn.Linear(2816, 128)
        self.output_layer = nn.Linear(128, 1)

        self.input_1 = nn.Linear(input_shape_1[0], 1024)
        self.self_attn_1 = SelfAttention(1024)
        self.transformer_1 = TransformerBlock(1024, 4)
        self.pooling_1 = nn.AdaptiveAvgPool1d(1)
        self.dense_1 = nn.Linear(1024, 1024)
        self.dropout_1 = nn.Dropout(p=0.05)

        self.input_2 = nn.Linear(input_shape_2[0], 1024)
        self.self_attn_2 = SelfAttention(1024)
        self.transformer_2 = TransformerBlock(1024, 4)
        self.pooling_2 = nn.AdaptiveAvgPool1d(1)
        self.dense_2 = nn.Linear(1024, 1024)
        self.dropout_2 = nn.Dropout(p=0.05)

        self.num_layers = num_layers

        self.convs1 = nn.ModuleList()
        self.convs1.append(GCNConv(4, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs1.append(GCNConv(hidden_channels, hidden_channels))

        self.convs2 = nn.ModuleList()
        self.convs2.append(GCNConv(4, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs2.append(GCNConv(hidden_channels, hidden_channels))

        self.cross_att = GATConv(hidden_channels, hidden_channels, heads=2)

        self.lin1 = nn.Linear(2816, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)

        self.transform = NormalizeScale()

    def forward(self, data_batch_1, data_batch_2):
        x1 = data_batch_1.x.double()
        edge_index_1 = data_batch_1.edge_index
        z1 = data_batch_1.z
        y1_coord = data_batch_1.y_coord
        z1_coord = data_batch_1.z_coord

        concatenated_x1 = torch.stack([x1, z1, y1_coord, z1_coord], dim=1)

        # Apply graph normalization
        data_batch_1 = self.transform(data_batch_1)
        x1 = data_batch_1.x

        for i in range(self.num_layers):
            concatenated_x1 = self.convs1[i](concatenated_x1, edge_index_1.to(torch.int64))
            concatenated_x1 = F.relu(concatenated_x1.double())

        # Process second graph
        x2 = data_batch_2.x
        edge_index_2 = data_batch_2.edge_index
        z2 = data_batch_2.z
        y2_coord = data_batch_2.y_coord
        z2_coord = data_batch_2.z_coord

        concatenated_x2 = torch.stack([x2, z2, y2_coord, z2_coord], dim=1)

        # Apply graph normalization
        data_batch_2 = self.transform(data_batch_2)
        x2 = data_batch_2.x

        for i in range(self.num_layers):
            concatenated_x2 = self.convs2[i](concatenated_x2, edge_index_2.to(torch.int64))
            concatenated_x2 = F.relu(concatenated_x2)

        # Cross-attention block
        x1 = self.cross_att(concatenated_x1, edge_index_1.to(torch.int64))
        x2 = self.cross_att(concatenated_x2, edge_index_2.to(torch.int64))

        # Concatenate all tensors along the last dimension
        x = torch.cat([
            global_mean_pool(x1, data_batch_1.batch),
            global_mean_pool(x2, data_batch_2.batch)], dim=1)

        input_11 = self.input_1(data_batch_1.seq)
        self_attn_1 = self.self_attn_1(input_11)
        transformer_1 = self.transformer_1(self_attn_1)
        pooling_1 = self.pooling_1(transformer_1.transpose(0, 1)).squeeze(dim=1)
        dense_1 = self.dense_1(pooling_1)
        dropout_1 = self.dropout_1(dense_1)

        input_22 = self.input_2(data_batch_2.seq)
        self_attn_2 = self.self_attn_2(input_22)
        transformer_2 = self.transformer_2(self_attn_2)
        pooling_2 = self.pooling_2(transformer_2.transpose(0, 1)).squeeze(dim=1)
        dense_2 = self.dense_2(pooling_2)
        dropout_2 = self.dropout_2(dense_2)

        input_shape = [(dropout_1.shape[-1],), (dropout_2.shape[-1],)]
        cross_attn_1 = self.cross_attn_1([dropout_1, dropout_2])
        cross_attn_2 = self.cross_attn_2([self.self_attn_1(input_11), self.self_attn_2(input_22)])
        cross_pooling = self.cross_pooling(cross_attn_2.transpose(0, 1)).squeeze(dim=1)
        cross_attn = self.cross_attn([cross_attn_1, cross_pooling])
        cross_atten = F.tanh(self.dense(cross_attn))

        self_atten_1 = self.self_atten_1(self_attn_1.transpose(0, 1)).squeeze(dim=1)
        self_atten_2 = self.self_atten_2(self_attn_2.transpose(0, 1)).squeeze(dim=1)
        attention_scores = torch.cat([self_atten_1, self_atten_2, cross_atten], dim=-1)

        x_2 = torch.cat([self.pooling_2(x.transpose(0, 1)).squeeze(dim=1), attention_scores], dim = -1)

        attention_scores = torch.cat([attention_scores, self.pooling_2(x.transpose(0, 1)).squeeze(dim=1)], dim = -1)

        output_layer1 = F.tanh(self.output_layer1(attention_scores)) #transformer
        output_layer = self.output_layer(output_layer1)

        x = F.relu(self.lin1(x_2))
        x = self.lin2(x)

        return x, output_layer

input_shape_1 = (int(max_antibody_sequence_length*20),)
input_shape_2 = (int(max_antigen_sequence_length*20),)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
com_model = CombinedModel().to(device)
com_model = com_model.to(torch.float64)

alpha = 0.45
beta = 0.55
gamma = 0.05

com_model_cp  = torch.load('./content/model_weights.pth')
com_model_epoch = com_model_cp['epoch']
com_model.load_state_dict(com_model_cp['model_state_dict'])

# Evaluate the model
com_model.eval()
test_loss = 0.0
test_mae = 0.0
total_samples = 0

with torch.no_grad():
    for input in test_loader:
        input_1 = input[0].to(device)
        input_2 = input[1].to(device)
        target = input[2].to(device)
        batch_size = input_1.size(0)

        if input_1.pos is not None:
            input_1 = NormalizeScale()(input_1)
        else:
            print("Data does not have position information, skipping normalization.")

        if input_2.pos is not None:
            input_2 = NormalizeScale()(input_2)
        else:
            print("Data does not have position information, skipping normalization.")

        output_gnn, output_tranf = com_model(input_1, input_2)
        print(f"Predicted binding affinity for the provided antibody-antigen pair (in IC50): {10**(alpha*output_gnn.item()+beta*output_tranf.item()+gamma*np.abs(output_gnn.item() - output_tranf.item()))}") #
        

