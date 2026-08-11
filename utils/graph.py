import os
import torch
import multiprocessing
from torch_geometric.data import HeteroData
import numpy as np
import json

design_list = ['adaptec1', 'adaptec2', 'adaptec3', 'adaptec4', 'bigblue1', 'bigblue2', 'bigblue3', 'bigblue4', ]


def read_nodes_file(file_path, io_list=None):
    """Reads a nodes file and returns a dictionary mapping node IDs to their id and size."""
    macro_list = []
    macro_dict = {}      # macro_name -> id & size
    macro_size_list = []

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    macro_idx = 0
    with open(file_path, 'r') as f:
        while True:
            line = f.readline()
            if not line: break
            words = line.split()
            if len(words) < 4: continue
            if words[-1] == 'terminal':
                macro_name = words[0]
                width = int(words[1])
                height = int(words[2])
                if io_list is None: 
                    macro_list.append(macro_name)
                    macro_dict[macro_name] = macro_idx
                    macro_size_list.append([width, height])
                    macro_idx += 1
                elif macro_name not in io_list:
                    macro_list.append(macro_name)
                    macro_dict[macro_name] = macro_idx
                    macro_size_list.append([width, height])
                    macro_idx += 1

    return macro_list, macro_dict, macro_size_list


def read_pl_file(file_path, macro_list):
    """Reads a placement file and returns a dictionary mapping node IDs to their positions."""
    macro_pos_list = [0] * len(macro_list)
    cell_idx = 0
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r') as f:
        while True:
            line = f.readline()
            if not line: break
            words = line.split()
            if len(words) < 5: continue
            #elif words[-1] == '/FIXED':
            else:
                macro_name = words[0]
                x = int(words[1])
                y = int(words[2])
                if macro_name in macro_list:
                    macro_pos_list[macro_list.index(macro_name)] = [x, y]
                    cell_idx += 1

    if cell_idx != (len(macro_list)):
        raise ValueError(f"Expected {len(macro_list)} macros, but found {cell_idx} in placement file.")

    return macro_pos_list


def read_nets_file(file_path, io_list, macro_list, ignore_degree=100):
    """Reads a nets file and returns a dictionary mapping net IDs to their nodes."""
    net_dict = {}  # net_id -> list of node ids
    net_degree_list = []
    src_cell_list = []
    dst_net_list = []
    attr_list = []

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    net_idx = 0
    with open(file_path, 'r') as f:
        while True:
            line = f.readline()
            if not line: break
            words = line.split()
            if len(words) < 4: continue
            elif words[0] == 'NetDegree':
                degree = int(words[2])
                net_name = words[3]
                temp_cell_list = []
                temp_cell_offset_list = []
                valid = False
                for _ in range(degree):
                    line = f.readline()
                    if not line: break
                    words = line.split()
                    cell_name = words[0]
                    if cell_name in io_list:
                        temp_cell_list.append(cell_name)
                        temp_cell_offset_list.append([float(words[3]), float(words[4])])
                    elif cell_name in macro_list:
                        temp_cell_list.append(cell_name)
                        temp_cell_offset_list.append([float(words[3]), float(words[4])])
                        valid = True
                if valid:
                    net_dict[net_name] = net_idx
                    net_degree_list.append([len(temp_cell_list), degree])
                    src_cell_list.extend(temp_cell_list)
                    dst_net_list.extend([net_idx] * len(temp_cell_list))
                    attr_list.extend(temp_cell_offset_list)
                    net_idx += 1

    np_io_list = np.array(io_list)
    np_src_cell_list = np.array(src_cell_list)
    mask = np.isin(np_io_list, np_src_cell_list)
    filtered_io_list = np_io_list[mask]

    cell_list = filtered_io_list.tolist() + macro_list
    src_cell_list = [cell_list.index(cell) for cell in src_cell_list]

    return net_dict, net_degree_list, src_cell_list, dst_net_list, attr_list, cell_list, mask


def read_scl_file(file_path):
    """Reads a scl file and returns maximum x and y coordinates."""
    max_height = 0
    max_width = 0

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r') as f:
        while True:
            line = f.readline()
            if not line: break
            words = line.split()
            if len(words) < 1: continue
            if words[0] == 'CoreRow':
                while True:
                    line = f.readline()
                    words = line.split()
                    if words[0] == 'End': break
                    elif words[0] == 'Coordinate': 
                        coordinate = int(words[2])
                    elif words[0] == 'Height':
                        max_height = max(max_height, coordinate+int(words[2]))
                    elif words[0] == 'SubrowOrigin': 
                        max_width = max(max_width, int(words[2])+int(words[5]))

    return max_height, max_width


def create_hetero_data(args):
    """Creates a HeteroData object."""
    
    (design, idx, pl_file, cell_list, size_list, degree_list, src_list, dst_list, edge_attr_list, max_height, max_width, num_io, num_macro) = args
    print(f'run {design} {idx}')

    pos_list = read_pl_file(pl_file, cell_list)

    data = HeteroData()
    data['cell']['pos'] = torch.tensor(pos_list)
    data['cell']['size'] = torch.tensor(size_list)
    data['net']['degree'] = torch.tensor(degree_list)
    data['cell', 'out', 'net'].edge_index = torch.cat([torch.tensor(src_list).view(1, -1), torch.tensor(dst_list).view(1, -1)], dim=0)
    data['cell', 'out', 'net'].edge_attr = torch.tensor(edge_attr_list)
    data['net', 'in', 'cell'].edge_index = torch.cat([torch.tensor(dst_list).view(1, -1), torch.tensor(src_list).view(1, -1)], dim=0)
    data['net', 'in', 'cell'].edge_attr = torch.tensor(edge_attr_list)
    data['num_io'] = num_io
    data['num_macro'] = num_macro
    data['max_size'] = torch.tensor([max_width, max_height])
    print(data['cell']['pos'].shape)

    torch.save(data, f'./dataset/test/{design}.pt')


def create_dataset(design_name, bench_path, place_path):
    io_nodes_file = os.path.join(bench_path, design_name, design_name + '.nodes')
    macro_nodes_file = os.path.join('./benchmarks/ispd2005', design_name, design_name + '.nodes')
    pl_file = os.path.join(bench_path, design_name, design_name + '.pl')
    nets_file = os.path.join(bench_path, design_name, design_name + '.nets')
    scl_file = os.path.join(bench_path, design_name, design_name + '.scl')

    io_list, io_dict, io_size_list = read_nodes_file(io_nodes_file)
    macro_list, macro_dict, macro_size_list = read_nodes_file(macro_nodes_file, io_list)
    net_dict, net_degree_list, src_cell_list, dst_net_list, edge_attr_list, cell_list, mask = read_nets_file(nets_file, io_list, macro_list)
    max_height, max_width = read_scl_file(scl_file)
    num_io = len(np.array(io_list)[mask].tolist())
    num_macro = len(macro_list)
    size_list = np.array(io_size_list)[mask].tolist() + macro_size_list

    with open(f'./map/{design_name}.json', 'w') as f:
        json.dump(cell_list, f)

    args_list = []
    for i in range(1):
        pl_file = os.path.join(place_path, design_name, design_name + f'_{i}', design_name + f'.gp.pl')
        args = (design_name, i, pl_file, cell_list, size_list, net_degree_list, src_cell_list, dst_net_list, edge_attr_list, max_height, max_width, num_io, num_macro)
        args_list.append(args)

    with multiprocessing.Pool(1) as pool:
        pool.map(create_hetero_data, args_list)


if __name__ == '__main__':
    bench_path = './benchmarks/mms'
    place_path = './dataset/placement_ref'
    dataset_path = './dataset_mms'

    data_dict = {}
    for design in design_list:
        create_dataset(design, bench_path, place_path)

        data_list = []
        for i in range(1):
            file_name = f'{design}_ref.pt'
            #print(f'Loading {file_name}')
            data = torch.load(f'./dataset/{design}/{file_name}')
            data_list.append(data)

        data_dict[design] = data_list
        print(f'Saved {len(data_list)} data items')

    torch.save(data_dict, f'{dataset_path}/test.pt')