from torch_geometric.nn import GATConv, SAGEConv, GCNConv, GraphConv, GINConv
from torch.nn import Linear
from .gat import DynamicsGATConv


def get_GATConv(in_channels, out_channels, config):
    return GATConv(
        in_channels,
        out_channels,
        heads=config.heads,
        concat=config.concat,
        bias=config.bias,
    )


def get_SAGEConv(in_channels, out_channels, config):
    if config.concat:
        out_channels *= config.heads
    return SAGEConv(in_channels, out_channels, bias=config.bias)


def get_GCNConv(in_channels, out_channels, config):
    if config.concat:
        out_channels *= config.heads
    return GCNConv(
        in_channels,
        out_channels,
        bias=config.bias,
    )


def get_MaxGraphConv(in_channels, out_channels, config):
    if config.concat:
        out_channels *= config.heads
    return GraphConv(in_channels, out_channels, bias=config.bias, aggr="max")


def get_MeanGraphConv(in_channels, out_channels, config):
    if config.concat:
        out_channels *= config.heads
    return GraphConv(in_channels, out_channels, bias=config.bias, aggr="mean")


def get_AddGraphConv(in_channels, out_channels, config):
    if config.concat:
        out_channels *= config.heads
    return GraphConv(in_channels, out_channels, bias=config.bias, aggr="add")


def get_DynamicsGATConv(in_channels, out_channels, config):
    return DynamicsGATConv(
        in_channels,
        out_channels,
        heads=config.heads,
        concat=config.concat,
        bias=config.bias,
        self_attention=config.self_attention,
    )


__dynamics__ = {
    "GATConv": get_GATConv,
    "SAGEConv": get_SAGEConv,
    "GCNConv": get_GCNConv,
    "AddGraphConv": get_AddGraphConv,
    "MeanGraphConv": get_MeanGraphConv,
    "MaxGraphConv": get_MaxGraphConv,
    "DynamicsGATConv": get_DynamicsGATConv,
}


def get(in_channels, out_channels, config):
    name = config.gnn_name
    if name in __dynamics__:
        return __dynamics__[name](in_channels, out_channels, config)
    else:
        raise ValueError(
            f"{name} is invalid, possible entries are {list(__dynamics__.keys())}"
        )
