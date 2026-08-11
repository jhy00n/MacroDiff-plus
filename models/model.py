import torch
import torch.nn as nn
import torch.nn.functional as F

from models.graph import GraphConv
from models.trans import TransConv


def pad_and_stack(sequences, max_len=None):
    if max_len == None:
        max_len = max(len(seq) for seq in sequences)
    padded = [torch.nn.functional.pad(seq, (0, 0, 0, max_len - len(seq)), value=0.) for seq in sequences]
    padded_tensor = torch.stack(padded)

    mask_list = []
    for seq in sequences:
        valid_len = len(seq)
        mask = torch.cat([
            torch.ones(valid_len, dtype=torch.bool, device=seq.device),
            torch.zeros(max_len - valid_len, dtype=torch.bool, device=seq.device)
        ])
        mask_list.append(mask)
    mask_tensor = torch.stack(mask_list)

    return padded_tensor, mask_tensor

def process_chunks(data, split_sizes, num_io, batch_size):
    chunks = []
    start = 0
    for i in range(batch_size):
        end = split_sizes[i]
        chunk = data[start:end]
        chunks.append((chunk[:num_io[i]], chunk[num_io[i]:]))
        start = end
    return chunks
    

class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1. / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        x = x.float().unsqueeze(-1)
        sinusoid_inp = x * self.inv_freq
        sin_emb = torch.sin(sinusoid_inp)
        cos_emb = torch.cos(sinusoid_inp)
        emb = torch.cat([sin_emb, cos_emb], dim=-1)
        return emb


class GraphPlacer(nn.Module):
    def __init__(
            self, 
            in_cell_channels, 
            in_net_channels, 
            hidden_channels, 
            edge_dim, 
            num_graph_layers=5, 
            num_heads=4, 
            dropout=0., 
            use_residual=True, 
            use_bn=True, 
        ):
        super().__init__()

        self.in_cell_channels = in_cell_channels
        self.in_net_channels = in_net_channels
        self.edge_dim = edge_dim
        self.hidden_channels = hidden_channels
        self.time_feats = hidden_channels * 4
        self.num_graph_layers = num_graph_layers
        self.dropout = dropout

        self.graph_conv = GraphConv(
            in_cell_channels, 
            in_net_channels, 
            hidden_channels, 
            edge_dim, 
            self.time_feats,
            num_layers=num_graph_layers, 
            num_heads=num_heads, 
            dropout=dropout, 
            use_bn=use_bn, 
            use_residual=use_residual,
        )

        self.cell_fc = nn.Linear(self.hidden_channels, 2)
        self.net_fc = nn.Linear(self.hidden_channels, 1)

    def forward(self, data, t_cell, t_net, *args, **kwargs):
        x_g = self.graph_conv(data, t_cell, t_net)
        
        x_cell = x_g['cell']
        x_cell = self.cell_fc(x_cell)
        
        x_net = x_g['net']
        x_net = self.net_fc(x_net)
        
        return x_net, x_cell


class TransPlacer(nn.Module):
    def __init__(
            self, 
            in_cell_channels, 
            hidden_channels, 
            edge_dim, 
            num_trans_layers=2,
            num_heads=4, 
            dropout=0.
        ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.time_feats = hidden_channels * 4
        self.dropout = dropout

        self.trans_conv = TransConv(
            in_channels=2,
            out_channels=2,
            condition_channels=2,
            time_emb_dim=self.time_feats,
            model_channels=32,
            channel_mults=(1, 2, 4, 8), 
            attention_levels=[1, 2],
            num_heads=num_heads,
            dropout=dropout
        )

    def forward(self, data, t_cell, t_net, cell_x, t_trans, cond, mask=None):
        x_t = self.trans_conv(cell_x, t_trans, cond, mask=mask)
        x_net = None
        
        return x_net, x_t


class MacroPlacer(nn.Module):
    def __init__(
            self, 
            in_cell_channels, 
            in_net_channels, 
            hidden_channels, 
            edge_dim, 
            num_graph_layers=5, 
            num_trans_layers=2,
            num_heads=4, 
            dropout=0., 
            use_residual=True, 
            use_bn=True, 
        ):
        super().__init__()

        self.in_cell_channels = in_cell_channels
        self.in_net_channels = in_net_channels
        self.edge_dim = edge_dim
        self.hidden_channels = hidden_channels
        self.time_feats = hidden_channels * 4
        self.num_graph_layers = num_graph_layers
        self.num_trans_layers = num_trans_layers
        self.dropout = dropout

        self.graph_conv = GraphConv(
            in_cell_channels, 
            in_net_channels, 
            hidden_channels, 
            edge_dim, 
            self.time_feats,
            num_layers=num_graph_layers, 
            num_heads=num_heads, 
            dropout=dropout, 
            use_bn=use_bn, 
            use_residual=use_residual,
        )

        self.trans_conv = TransConv(
            in_channels=hidden_channels,
            out_channels=2,
            condition_channels=2,
            time_emb_dim=self.time_feats,
            model_channels=32,
            channel_mults=(1, 2, 4, 8), 
            attention_levels=[1, 2],
            num_heads=num_heads,
            dropout=dropout
        )

        self.net_fc = nn.Linear(self.hidden_channels, 1)

    def forward(self, data, t_cell, t_net, t_trans, cond, batch_size, max_macro, split_sizes=None, num_io=None, mask=None):
        x_g = self.graph_conv(data, t_cell, t_net)
        x_net = x_g['net']
        x_net = self.net_fc(x_net)

        x_cell = x_g['cell']
        if batch_size > 1:
            cell_chunks = process_chunks(x_cell, split_sizes, num_io, batch_size)
            _, x_macro_list = zip(*cell_chunks)
            x_macro_pad, _ = pad_and_stack(x_macro_list, max_macro)
            x_t = self.trans_conv(x_macro_pad, t_trans, cond, mask=mask)
        else:
            x_t = self.trans_conv(x_cell[-max_macro:].unsqueeze(0), t_trans, cond, mask=mask)

        return x_net, x_t
