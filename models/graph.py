import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

from models.trans import SinusoidalPositionEmbeddings

torch.set_printoptions(profile="full")

class GraphConv(nn.Module):
    def __init__(
            self, 
            in_cell_channels, 
            in_net_channels, 
            hidden_channels, 
            edge_dim, 
            time_emb_dim, 
            num_layers=5, 
            num_heads=1, 
            dropout=0.0, 
            use_bn=True, 
            use_residual=True,
    ):
        super().__init__()

        self.in_cell_channels = in_cell_channels
        self.in_net_channels = in_net_channels
        self.edge_dim = edge_dim
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_bn = use_bn
        self.use_residual = use_residual

        self.activation = F.elu

        self.time_embedder = SinusoidalPositionEmbeddings(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.GELU(),
            nn.Linear(time_emb_dim * 4, hidden_channels)
        )

        self.cell_fcs = nn.ModuleList()
        self.net_fcs = nn.ModuleList()
        self.cell_bns = nn.ModuleList()
        self.net_bns = nn.ModuleList()
        self.convs = nn.ModuleList()

        self.cell_fcs.append(nn.Linear(in_cell_channels, hidden_channels))
        self.net_fcs.append(nn.Linear(in_net_channels, hidden_channels))
        self.cell_bns.append(nn.LayerNorm(hidden_channels))
        self.net_bns.append(nn.LayerNorm(hidden_channels))

        for i in range(num_layers):
            self.convs.append(
                gnn.HeteroConv({
                    ('cell', 'out', 'net'): gnn.GATv2Conv(
                        hidden_channels, 
                        hidden_channels, 
                        edge_dim=edge_dim, 
                        heads=num_heads, 
                        dropout=dropout, 
                        add_self_loops=False, 
                        concat=False,
                    ),
                    ('net', 'in', 'cell'): gnn.GATv2Conv(
                        hidden_channels, 
                        hidden_channels, 
                        edge_dim=edge_dim, 
                        heads=num_heads, 
                        dropout=dropout, 
                        add_self_loops=False, 
                        concat=False,
                    ),
                }, aggr='mean')
            )
            self.cell_bns.append(nn.LayerNorm(hidden_channels))
            self.net_bns.append(nn.LayerNorm(hidden_channels))

    def reset_parameters(self):
        for fc in self.cell_fcs:
            fc.reset_parameters()
        for fc in self.net_fcs:
            fc.reset_parameters()
        for bn in self.cell_bns:
            bn.reset_parameters()
        for bn in self.net_bns:
            bn.reset_parameters()
        for hetero_conv in self.convs:
            hetero_conv.reset_parameters()

    def forward(self, data, t_cell, t_net, cell_emb=None, net_emb=None):
        t_cell_emb = self.time_embedder(t_cell)
        t_cell_emb = self.time_mlp(t_cell_emb)

        t_net_emb = self.time_embedder(t_net)
        t_net_emb = self.time_mlp(t_net_emb)

        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        edge_attr_dict = data.edge_attr_dict

        x_dict['cell'] = self.cell_fcs[0](x_dict['cell'])
        x_dict['net'] = self.net_fcs[0](x_dict['net'])

        if self.use_bn:
            x_dict['cell'] = self.cell_bns[0](x_dict['cell'])
            x_dict['net'] = self.net_bns[0](x_dict['net'])
        x_dict['cell'] = self.activation(x_dict['cell'])
        x_dict['net'] = self.activation(x_dict['net'])
        x_dict['cell'] = F.dropout(x_dict['cell'], p=self.dropout, training=self.training)
        x_dict['net'] = F.dropout(x_dict['net'], p=self.dropout, training=self.training)

        prev_layer = x_dict

        for i, conv in enumerate(self.convs):
            prev_cell = prev_layer["cell"]
            prev_net = prev_layer["net"]

            x_dict['cell'] = x_dict['cell'] + t_cell_emb
            if cell_emb is not None:
                x_dict['cell'] = x_dict['cell'] + cell_emb
            x_dict['net'] = x_dict['net'] + t_net_emb
            if net_emb is not None:
                x_dict['net'] = x_dict['net'] + net_emb

            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            if self.use_residual:
                x_dict['cell'] = x_dict['cell'] + prev_cell
                x_dict['net'] = x_dict['net'] + prev_net
            if self.use_bn:
                x_dict['cell'] = self.cell_bns[i+1](x_dict['cell'])
                x_dict['net'] = self.net_bns[i+1](x_dict['net'])
            x_dict['cell'] = self.activation(x_dict['cell'])
            x_dict['net'] = self.activation(x_dict['net'])
            x_dict['cell'] = F.dropout(x_dict['cell'], p=self.dropout, training=self.training)
            x_dict['net'] = F.dropout(x_dict['net'], p=self.dropout, training=self.training)

            prev_layer = x_dict

        return x_dict