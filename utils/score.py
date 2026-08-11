import torch
import torch_scatter
import torch.nn.functional as F


def get_hpwl_tensor(data, pos_tensor):
    cell_pos_per_edge = pos_tensor[data['cell', 'out', 'net'].edge_index[0]]
    offset_list = data['cell', 'out', 'net'].edge_attr
    cell_pos_with_offset = cell_pos_per_edge + offset_list
    net_cell_ids = data['cell', 'out', 'net'].edge_index[1]

    x_min = torch_scatter.scatter(cell_pos_with_offset[:,0], net_cell_ids, reduce='min')
    x_max = torch_scatter.scatter(cell_pos_with_offset[:,0], net_cell_ids, reduce='max')
    y_min = torch_scatter.scatter(cell_pos_with_offset[:,1], net_cell_ids, reduce='min')
    y_max = torch_scatter.scatter(cell_pos_with_offset[:,1], net_cell_ids, reduce='max')

    width = x_max - x_min
    height = y_max - y_min
    hpwl_list = torch.stack((width, height), dim=1)
    hpwl = hpwl_list.sum()

    return hpwl, hpwl_list


def get_overlap_tensor(size_tensor, pos_tensor):
    cell_mins = pos_tensor
    cell_maxs = pos_tensor + size_tensor

    left = cell_mins[:, 0]
    right = cell_maxs[:, 0]
    bottom = cell_mins[:, 1]
    top = cell_maxs[:, 1]

    left_overlap = torch.maximum(left[:, None], left)
    right_overlap = torch.minimum(right[:, None], right)
    bottom_overlap = torch.maximum(bottom[:, None], bottom)
    top_overlap = torch.minimum(top[:, None], top)

    overlap_width = torch.clamp(right_overlap - left_overlap, min=0)
    overlap_height = torch.clamp(top_overlap - bottom_overlap, min=0)
    overlap_area = overlap_width * overlap_height
    self_area = size_tensor[:, 0] * size_tensor[:, 1]

    overlap_cost = (torch.sum(overlap_area) - torch.sum(self_area))
    overlap_list = (torch.sum(overlap_area, dim=1) - self_area)

    return overlap_cost, overlap_list


def get_hpwl_diff_test(data, pos_tensor, gamma=0.01, eps=1e-12):

    edge_index = data['cell', 'out', 'net'].edge_index
    offset_list = data['cell', 'out', 'net']['offset']
    net_cell_ids = edge_index[1]  # 2307
    cell_indices = edge_index[0]  # 2307
    cell_pos = pos_tensor[cell_indices]
    cell_pos_with_offset = cell_pos + offset_list

    x = cell_pos_with_offset[:,0]
    y = cell_pos_with_offset[:,1]

    max_x = torch_scatter.scatter(x, net_cell_ids, reduce='max')
    min_x = torch_scatter.scatter(x, net_cell_ids, reduce='min')
    x_diff_max = x - max_x[net_cell_ids]
    x_diff_min = x - min_x[net_cell_ids]
    exp_max_x = torch.exp(x_diff_max / gamma)
    exp_min_x = torch.exp(-x_diff_min / gamma)
    sum_exp_max_x = torch_scatter.scatter(exp_max_x, net_cell_ids, reduce='sum')
    sum_x_exp_max_x = torch_scatter.scatter(x_diff_max * exp_max_x, net_cell_ids, reduce='sum')
    sum_exp_min_x = torch_scatter.scatter(exp_min_x, net_cell_ids, reduce='sum')
    sum_x_exp_min_x = torch_scatter.scatter(x_diff_min * exp_min_x, net_cell_ids, reduce='sum')
    wa_x_plus = max_x + sum_x_exp_max_x / (sum_exp_max_x + eps)
    wa_x_minus = min_x + sum_x_exp_min_x / (sum_exp_min_x + eps)
    wa_x = wa_x_plus - wa_x_minus

    max_y = torch_scatter.scatter(y, net_cell_ids, reduce='max')
    min_y = torch_scatter.scatter(y, net_cell_ids, reduce='min')
    y_diff_max = y - max_y[net_cell_ids]
    y_diff_min = y - min_y[net_cell_ids]
    exp_max_y = torch.exp(y_diff_max / gamma)
    exp_min_y = torch.exp(-y_diff_min / gamma)
    sum_exp_max_y = torch_scatter.scatter(exp_max_y, net_cell_ids, reduce='sum')
    sum_y_exp_max_y = torch_scatter.scatter(y_diff_max * exp_max_y, net_cell_ids, reduce='sum')
    sum_exp_min_y = torch_scatter.scatter(exp_min_y, net_cell_ids, reduce='sum')
    sum_y_exp_min_y = torch_scatter.scatter(y_diff_min * exp_min_y, net_cell_ids, reduce='sum')
    wa_y_plus = max_y + sum_y_exp_max_y / (sum_exp_max_y + eps)
    wa_y_minus = min_y + sum_y_exp_min_y / (sum_exp_min_y + eps)
    wa_y = wa_y_plus - wa_y_minus

    hpwl_list = wa_x + wa_y
    hpwl_list = hpwl_list.unsqueeze(-1)
    hpwl_approx = hpwl_list.sum()

    return hpwl_approx, hpwl_list


def get_hpwl_diff(data, pos_tensor, gamma=0.01, eps=1e-12):
    edge_index = data['cell', 'out', 'net'].edge_index
    offset_list = data['cell', 'out', 'net']['offset']
    cell_split_sizes = torch.cumsum(data['num_io'] + data['num_macro'], dim=0)
    net_split_sizes = torch.cumsum(data['num_net'], dim=0)
    pin_split_sizes = torch.cumsum(data['num_edge'], dim=0)
    length = data['leng']
    start = 0 
    hpwl_list_list = []
    hpwl_approx_list = []

    for i in range(len(pos_tensor)):
        end = pin_split_sizes[i].item()
        net_cell_ids = edge_index[1][start:end]
        if i == 0:
            net_cell_ids = net_cell_ids
        else:
            net_cell_ids = net_cell_ids-net_split_sizes[i-1]
        cell_indices = edge_index[0][start:end]
        if i == 0:
            cell_indices = cell_indices
        else:
            cell_indices = cell_indices-cell_split_sizes[i-1]
        cell_pos = pos_tensor[i][:length[i]]
        cell_pos = cell_pos[cell_indices]
        offset = offset_list[start:end]
        cell_pos_with_offset = cell_pos + offset
        
        x = cell_pos_with_offset[:,0]
        y = cell_pos_with_offset[:,1]
        max_x = torch_scatter.scatter(x, net_cell_ids, reduce='max')
        min_x = torch_scatter.scatter(x, net_cell_ids, reduce='min')
        x_diff_max = x - max_x[net_cell_ids]
        x_diff_min = x - min_x[net_cell_ids]
        exp_max_x = torch.exp(x_diff_max / gamma)
        exp_min_x = torch.exp(-x_diff_min / gamma)
        sum_exp_max_x = torch_scatter.scatter(exp_max_x, net_cell_ids, reduce='sum')
        sum_x_exp_max_x = torch_scatter.scatter(x_diff_max * exp_max_x, net_cell_ids, reduce='sum')
        sum_exp_min_x = torch_scatter.scatter(exp_min_x, net_cell_ids, reduce='sum')
        sum_x_exp_min_x = torch_scatter.scatter(x_diff_min * exp_min_x, net_cell_ids, reduce='sum')
        wa_x_plus = max_x + sum_x_exp_max_x / (sum_exp_max_x + eps)
        wa_x_minus = min_x + sum_x_exp_min_x / (sum_exp_min_x + eps)
        wa_x = wa_x_plus - wa_x_minus

        max_y = torch_scatter.scatter(y, net_cell_ids, reduce='max')
        min_y = torch_scatter.scatter(y, net_cell_ids, reduce='min')
        y_diff_max = y - max_y[net_cell_ids]
        y_diff_min = y - min_y[net_cell_ids]
        exp_max_y = torch.exp(y_diff_max / gamma)
        exp_min_y = torch.exp(-y_diff_min / gamma)
        sum_exp_max_y = torch_scatter.scatter(exp_max_y, net_cell_ids, reduce='sum')
        sum_y_exp_max_y = torch_scatter.scatter(y_diff_max * exp_max_y, net_cell_ids, reduce='sum')
        sum_exp_min_y = torch_scatter.scatter(exp_min_y, net_cell_ids, reduce='sum')
        sum_y_exp_min_y = torch_scatter.scatter(y_diff_min * exp_min_y, net_cell_ids, reduce='sum')
        wa_y_plus = max_y + sum_y_exp_max_y / (sum_exp_max_y + eps)
        wa_y_minus = min_y + sum_y_exp_min_y / (sum_exp_min_y + eps)
        wa_y = wa_y_plus - wa_y_minus

        hpwl_list = wa_x + wa_y
        hpwl_list = hpwl_list.unsqueeze(-1)
        hpwl_approx = hpwl_list.sum()

        hpwl_list_list.append(hpwl_list)
        hpwl_approx_list.append(hpwl_approx) 
        start = end

    return hpwl_approx_list, hpwl_list_list


def get_hpwl_norm_diff_test(data, pos_tensor, gamma=0.01, eps=1e-12):

    edge_index = data['cell', 'out', 'net'].edge_index
    offset_list = data['cell', 'out', 'net'].edge_attr
    net_cell_ids = edge_index[1]  # 2307
    cell_indices = edge_index[0]  # 2307
    cell_pos = pos_tensor[cell_indices]
    cell_pos_with_offset = cell_pos + offset_list

    x = cell_pos_with_offset[:,0]
    y = cell_pos_with_offset[:,1]

    max_x = torch_scatter.scatter(x, net_cell_ids, reduce='max')
    min_x = torch_scatter.scatter(x, net_cell_ids, reduce='min')
    x_diff_max = x - max_x[net_cell_ids]
    x_diff_min = x - min_x[net_cell_ids]
    exp_max_x = torch.exp(x_diff_max / gamma)
    exp_min_x = torch.exp(-x_diff_min / gamma)
    sum_exp_max_x = torch_scatter.scatter(exp_max_x, net_cell_ids, reduce='sum')
    sum_x_exp_max_x = torch_scatter.scatter(x_diff_max * exp_max_x, net_cell_ids, reduce='sum')
    sum_exp_min_x = torch_scatter.scatter(exp_min_x, net_cell_ids, reduce='sum')
    sum_x_exp_min_x = torch_scatter.scatter(x_diff_min * exp_min_x, net_cell_ids, reduce='sum')
    wa_x_plus = max_x + sum_x_exp_max_x / (sum_exp_max_x + eps)
    wa_x_minus = min_x + sum_x_exp_min_x / (sum_exp_min_x + eps)
    wa_x = wa_x_plus - wa_x_minus

    max_y = torch_scatter.scatter(y, net_cell_ids, reduce='max')
    min_y = torch_scatter.scatter(y, net_cell_ids, reduce='min')
    y_diff_max = y - max_y[net_cell_ids]
    y_diff_min = y - min_y[net_cell_ids]
    exp_max_y = torch.exp(y_diff_max / gamma)
    exp_min_y = torch.exp(-y_diff_min / gamma)
    sum_exp_max_y = torch_scatter.scatter(exp_max_y, net_cell_ids, reduce='sum')
    sum_y_exp_max_y = torch_scatter.scatter(y_diff_max * exp_max_y, net_cell_ids, reduce='sum')
    sum_exp_min_y = torch_scatter.scatter(exp_min_y, net_cell_ids, reduce='sum')
    sum_y_exp_min_y = torch_scatter.scatter(y_diff_min * exp_min_y, net_cell_ids, reduce='sum')
    wa_y_plus = max_y + sum_y_exp_max_y / (sum_exp_max_y + eps)
    wa_y_minus = min_y + sum_y_exp_min_y / (sum_exp_min_y + eps)
    wa_y = wa_y_plus - wa_y_minus

    hpwl_list = wa_x + wa_y
    hpwl_list = hpwl_list.unsqueeze(-1)
    hpwl_approx = hpwl_list.sum()

    return hpwl_approx, hpwl_list


def get_hpwl_norm_diff(data, pos_tensor, gamma=0.01, eps=1e-12):
    edge_index = data['cell', 'out', 'net'].edge_index
    offset_list = data['cell', 'out', 'net'].edge_attr
    cell_split_sizes = torch.cumsum(data['num_io'] + data['num_macro'], dim=0)
    net_split_sizes = torch.cumsum(data['num_net'], dim=0)
    pin_split_sizes = torch.cumsum(data['num_edge'], dim=0)
    length = data['leng']
    start = 0 
    hpwl_list_list = []
    hpwl_approx_list = []

    for i in range(len(pos_tensor)):
        end = pin_split_sizes[i].item()
        net_cell_ids = edge_index[1][start:end]
        if i == 0:
            net_cell_ids = net_cell_ids
        else:
            net_cell_ids = net_cell_ids-net_split_sizes[i-1]
        cell_indices = edge_index[0][start:end]
        if i == 0:
            cell_indices = cell_indices
        else:
            cell_indices = cell_indices-cell_split_sizes[i-1]
        cell_pos = pos_tensor[i][:length[i]]
        cell_pos = cell_pos[cell_indices]
        offset = offset_list[start:end]
        cell_pos_with_offset = cell_pos + offset
        
        x = cell_pos_with_offset[:,0]
        y = cell_pos_with_offset[:,1]
        max_x = torch_scatter.scatter(x, net_cell_ids, reduce='max')
        min_x = torch_scatter.scatter(x, net_cell_ids, reduce='min')
        x_diff_max = x - max_x[net_cell_ids]
        x_diff_min = x - min_x[net_cell_ids]
        exp_max_x = torch.exp(x_diff_max / gamma)
        exp_min_x = torch.exp(-x_diff_min / gamma)
        sum_exp_max_x = torch_scatter.scatter(exp_max_x, net_cell_ids, reduce='sum')
        sum_x_exp_max_x = torch_scatter.scatter(x_diff_max * exp_max_x, net_cell_ids, reduce='sum')
        sum_exp_min_x = torch_scatter.scatter(exp_min_x, net_cell_ids, reduce='sum')
        sum_x_exp_min_x = torch_scatter.scatter(x_diff_min * exp_min_x, net_cell_ids, reduce='sum')
        wa_x_plus = max_x + sum_x_exp_max_x / (sum_exp_max_x + eps)
        wa_x_minus = min_x + sum_x_exp_min_x / (sum_exp_min_x + eps)
        wa_x = wa_x_plus - wa_x_minus

        max_y = torch_scatter.scatter(y, net_cell_ids, reduce='max')
        min_y = torch_scatter.scatter(y, net_cell_ids, reduce='min')
        y_diff_max = y - max_y[net_cell_ids]
        y_diff_min = y - min_y[net_cell_ids]
        exp_max_y = torch.exp(y_diff_max / gamma)
        exp_min_y = torch.exp(-y_diff_min / gamma)
        sum_exp_max_y = torch_scatter.scatter(exp_max_y, net_cell_ids, reduce='sum')
        sum_y_exp_max_y = torch_scatter.scatter(y_diff_max * exp_max_y, net_cell_ids, reduce='sum')
        sum_exp_min_y = torch_scatter.scatter(exp_min_y, net_cell_ids, reduce='sum')
        sum_y_exp_min_y = torch_scatter.scatter(y_diff_min * exp_min_y, net_cell_ids, reduce='sum')
        wa_y_plus = max_y + sum_y_exp_max_y / (sum_exp_max_y + eps)
        wa_y_minus = min_y + sum_y_exp_min_y / (sum_exp_min_y + eps)
        wa_y = wa_y_plus - wa_y_minus

        hpwl_list = wa_x + wa_y
        hpwl_list = hpwl_list.unsqueeze(-1)
        hpwl_approx = hpwl_list.sum()

        hpwl_list_list.append(hpwl_list)
        hpwl_approx_list.append(hpwl_approx) 
        start = end

    return hpwl_approx_list, hpwl_list_list


def get_hpwl_grad(data, pos_tensor, gamma=0.01, eps=1e-12):
    edge_index = data['cell', 'out', 'net'].edge_index # 2, 2307
    offset_list = data['cell', 'out', 'net'].edge_attr # 2307, 2

    cell_split_sizes = torch.cumsum(data['num_io'] + data['num_macro'], dim=0) # 6, 63
    net_split_sizes = torch.cumsum(data['num_net'], dim=0) # 629
    pin_split_sizes = torch.cumsum(data['num_edge'], dim=0) # 2307
    length = data['leng']
    # pos_tensor 2813  # max_macro 959 max_io 1854
    start = 0 
    grad_list = []

    for i in range(len(pos_tensor)):
        end = pin_split_sizes[i].item()

        net_cell_ids = edge_index[1][start:end] # 2307
        if i > 0:
            net_cell_ids = net_cell_ids-net_split_sizes[i-1]

        cell_indices = edge_index[0][start:end] # 2307
        if i > 0:
            cell_indices = cell_indices-cell_split_sizes[i-1]

        cell_pos = pos_tensor[i][:length[i]] # 69 2
        cell_pos_pins = cell_pos[cell_indices] # 2307 2
        offset = offset_list[start:end]  # 2307 2
        pin_pos = cell_pos_pins + offset
        
        # x, y
        x = pin_pos[:,0]
        y = pin_pos[:,1]

        # 1) coarse max/min (scatter)
        max_x = torch_scatter.scatter(x, net_cell_ids, reduce='max')
        min_x = torch_scatter.scatter(x, net_cell_ids, reduce='min')
        max_y = torch_scatter.scatter(y, net_cell_ids, reduce='max')
        min_y = torch_scatter.scatter(y, net_cell_ids, reduce='min')

        # 2) diff
        x_diff_max = x - max_x[net_cell_ids]
        x_diff_min = x - min_x[net_cell_ids]
        y_diff_max = y - max_y[net_cell_ids]
        y_diff_min = y - min_y[net_cell_ids]

        # 3) exp(...)
        exp_max_x = torch.exp(x_diff_max / gamma)
        exp_min_x = torch.exp(-x_diff_min / gamma)
        exp_max_y = torch.exp(y_diff_max / gamma)
        exp_min_y = torch.exp(-y_diff_min / gamma)

        # 4) sum_exp, sum_x_exp
        sum_exp_max_x = torch_scatter.scatter(exp_max_x, net_cell_ids, reduce='sum')
        sum_x_exp_max_x = torch_scatter.scatter(x_diff_max * exp_max_x, net_cell_ids, reduce='sum')
        sum_exp_min_x = torch_scatter.scatter(exp_min_x, net_cell_ids, reduce='sum')
        sum_x_exp_min_x = torch_scatter.scatter(x_diff_min * exp_min_x, net_cell_ids, reduce='sum')

        sum_exp_max_y = torch_scatter.scatter(exp_max_y, net_cell_ids, reduce='sum')
        sum_y_exp_max_y = torch_scatter.scatter(y_diff_max * exp_max_y, net_cell_ids, reduce='sum')
        sum_exp_min_y = torch_scatter.scatter(exp_min_y, net_cell_ids, reduce='sum')
        sum_y_exp_min_y = torch_scatter.scatter(y_diff_min * exp_min_y, net_cell_ids, reduce='sum')

        dbplus_x = exp_max_x * (1.0 + x_diff_max / gamma)
        dSplus_x = (1.0 / gamma) * exp_max_x
        bplus_x_pin = sum_x_exp_max_x[net_cell_ids]
        Splus_x_pin = sum_exp_max_x[net_cell_ids]
        px_plus = (dbplus_x * Splus_x_pin - bplus_x_pin * dSplus_x) / ((Splus_x_pin + eps)**2)

        dbminus_x = exp_min_x * (1.0 - x_diff_min / gamma)
        dSminus_x = -(1.0 / gamma) * exp_min_x
        bminus_x_pin = sum_x_exp_min_x[net_cell_ids]
        Sminus_x_pin = sum_exp_min_x[net_cell_ids]
        px_minus = (dbminus_x * Sminus_x_pin - bminus_x_pin * dSminus_x) / ((Sminus_x_pin + eps)**2)

        grad_x_pins = px_plus - px_minus

        dbplus_y = exp_max_y * (1.0 + y_diff_max / gamma)
        dSplus_y = (1.0 / gamma) * exp_max_y
        bplus_y_pin = sum_y_exp_max_y[net_cell_ids]
        Splus_y_pin = sum_exp_max_y[net_cell_ids]
        py_plus = (dbplus_y * Splus_y_pin - bplus_y_pin * dSplus_y) / ((Splus_y_pin + eps)**2)
        
        dbminus_y = exp_min_y * (1.0 - y_diff_min / gamma)
        dSminus_y = -(1.0 / gamma) * exp_min_y
        bminus_y_pin = sum_y_exp_min_y[net_cell_ids]
        Sminus_y_pin = sum_exp_min_y[net_cell_ids]
        py_minus = (dbminus_y * Sminus_y_pin - bminus_y_pin * dSminus_y) / ((Sminus_y_pin + eps)**2)

        grad_y_pins = py_plus - py_minus

        grad_pin = torch.stack([grad_x_pins, grad_y_pins], dim=1)

        num_cell = length[i]
        num_net = data['num_net'][i].item()

        cell_net_grad = torch.zeros(num_cell * num_net, 2, device=cell_pos.device)
        flat_index = cell_indices.long() * num_net + net_cell_ids.long()
        cell_net_grad = torch_scatter.scatter_add(src=grad_pin, index=flat_index, dim=0, out=cell_net_grad)
        cell_net_grad = cell_net_grad.view(num_cell, num_net, 2)
        cell_net_grad = cell_net_grad.permute(0, 2, 1)
        
        grad_list.append(cell_net_grad)
        start = end

    return grad_list


def get_hpwl_grad_test(data, pos_tensor, gamma=0.01, eps=1e-12):
    edge_index = data['cell', 'out', 'net'].edge_index # 2, 2307
    offset_list = data['cell', 'out', 'net'].edge_attr # 2307, 2
    net_cell_ids = edge_index[1]  # 2307
    cell_indices = edge_index[0]  # 2307
    cell_pos = pos_tensor[cell_indices]
    cell_pos_with_offset = cell_pos + offset_list
        
    # x, y
    x = cell_pos_with_offset[:,0]
    y = cell_pos_with_offset[:,1]

    # 1) coarse max/min (scatter)
    max_x = torch_scatter.scatter(x, net_cell_ids, reduce='max')
    min_x = torch_scatter.scatter(x, net_cell_ids, reduce='min')
    max_y = torch_scatter.scatter(y, net_cell_ids, reduce='max')
    min_y = torch_scatter.scatter(y, net_cell_ids, reduce='min')

    # 2) diff
    x_diff_max = x - max_x[net_cell_ids]
    x_diff_min = x - min_x[net_cell_ids]
    y_diff_max = y - max_y[net_cell_ids]
    y_diff_min = y - min_y[net_cell_ids]

    # 3) exp(...)
    exp_max_x = torch.exp(x_diff_max / gamma)
    exp_min_x = torch.exp(-x_diff_min / gamma)
    exp_max_y = torch.exp(y_diff_max / gamma)
    exp_min_y = torch.exp(-y_diff_min / gamma)

    # 4) sum_exp, sum_x_exp
    sum_exp_max_x = torch_scatter.scatter(exp_max_x, net_cell_ids, reduce='sum')
    sum_x_exp_max_x = torch_scatter.scatter(x_diff_max * exp_max_x, net_cell_ids, reduce='sum')
    sum_exp_min_x = torch_scatter.scatter(exp_min_x, net_cell_ids, reduce='sum')
    sum_x_exp_min_x = torch_scatter.scatter(x_diff_min * exp_min_x, net_cell_ids, reduce='sum')

    sum_exp_max_y = torch_scatter.scatter(exp_max_y, net_cell_ids, reduce='sum')
    sum_y_exp_max_y = torch_scatter.scatter(y_diff_max * exp_max_y, net_cell_ids, reduce='sum')
    sum_exp_min_y = torch_scatter.scatter(exp_min_y, net_cell_ids, reduce='sum')
    sum_y_exp_min_y = torch_scatter.scatter(y_diff_min * exp_min_y, net_cell_ids, reduce='sum')

    dbplus_x = exp_max_x * (1.0 + x_diff_max / gamma)
    dSplus_x = (1.0 / gamma) * exp_max_x
    bplus_x_pin = sum_x_exp_max_x[net_cell_ids]
    Splus_x_pin = sum_exp_max_x[net_cell_ids]
    px_plus = (dbplus_x * Splus_x_pin - bplus_x_pin * dSplus_x) / ((Splus_x_pin + eps)**2)

    dbminus_x = exp_min_x * (1.0 - x_diff_min / gamma)
    dSminus_x = -(1.0 / gamma) * exp_min_x
    bminus_x_pin = sum_x_exp_min_x[net_cell_ids]
    Sminus_x_pin = sum_exp_min_x[net_cell_ids]
    px_minus = (dbminus_x * Sminus_x_pin - bminus_x_pin * dSminus_x) / ((Sminus_x_pin + eps)**2)

    grad_x_pins = px_plus - px_minus

    dbplus_y = exp_max_y * (1.0 + y_diff_max / gamma)
    dSplus_y = (1.0 / gamma) * exp_max_y
    bplus_y_pin = sum_y_exp_max_y[net_cell_ids]
    Splus_y_pin = sum_exp_max_y[net_cell_ids]
    py_plus = (dbplus_y * Splus_y_pin - bplus_y_pin * dSplus_y) / ((Splus_y_pin + eps)**2)
        
    dbminus_y = exp_min_y * (1.0 - y_diff_min / gamma)
    dSminus_y = -(1.0 / gamma) * exp_min_y
    bminus_y_pin = sum_y_exp_min_y[net_cell_ids]
    Sminus_y_pin = sum_exp_min_y[net_cell_ids]
    py_minus = (dbminus_y * Sminus_y_pin - bminus_y_pin * dSminus_y) / ((Sminus_y_pin + eps)**2)

    grad_y_pins = py_plus - py_minus

    grad_pin = torch.stack([grad_x_pins, grad_y_pins], dim=1)

    num_cell = data['num_io'] + data['num_macro']
    num_net = data['num_net']

    cell_net_grad = torch.zeros(num_cell * num_net, 2, device=cell_pos.device)
    flat_index = cell_indices.long() * num_net + net_cell_ids.long()
    cell_net_grad = torch_scatter.scatter_add(src=grad_pin, index=flat_index, dim=0, out=cell_net_grad)
    cell_net_grad = cell_net_grad.view(num_cell, num_net, 2)
    cell_net_grad = cell_net_grad.permute(0, 2, 1)

    return cell_net_grad


def get_overlap_diff(size_tensor, pos_tensor, gamma=0.01):
    def smooth_max(a, b):
        max_val = torch.max(a, b)
        return gamma * (torch.log(torch.exp((a - max_val) / gamma) + torch.exp((b - max_val) / gamma))) + max_val

    def smooth_min(a, b):
        min_val = torch.min(a, b)
        return -gamma * (torch.log(torch.exp((-a + min_val) / gamma) + torch.exp((-b + min_val) / gamma))) + min_val
    
    #pos = pos_tensor.clone().detach().requires_grad_(True)

    cell_mins = pos_tensor
    cell_maxs = pos_tensor + size_tensor

    left    = cell_mins[:, 0]
    right   = cell_maxs[:, 0]
    bottom  = cell_mins[:, 1]
    top     = cell_maxs[:, 1]

    left_overlap    = smooth_max(left[:, None], left)
    right_overlap   = smooth_min(right[:, None], right)
    bottom_overlap  = smooth_max(bottom[:, None], bottom)
    top_overlap     = smooth_min(top[:, None], top)

    overlap_width   = torch.clamp(right_overlap - left_overlap, min=0)
    overlap_height  = torch.clamp(top_overlap - bottom_overlap, min=0)
    overlap_area    = overlap_width * overlap_height

    self_mask       = torch.eye(overlap_area.size(0), dtype=torch.bool, device=overlap_area.device)
    overlap_area    = overlap_area.masked_fill(self_mask, 0.)

    overlap_cost    = torch.sum(overlap_area)
    overlap_list    = torch.sum(overlap_area, dim=1)

    return overlap_cost, overlap_list


def get_overlap_force(size_tensor, pos_tensor, gamma=0.01):
    """
    Calculates a differentiable repulsion force based on a smooth overlap area.
    This version is fully differentiable and avoids hard masks.
    """
    def smooth_max(a, b):
        max_val = torch.max(a, b)
        return gamma * torch.log(torch.exp((a - max_val) / gamma) + torch.exp((b - max_val) / gamma)) + max_val

    def smooth_min(a, b):
        min_val = torch.min(a, b)
        return -gamma * (torch.log(torch.exp((-a + min_val) / gamma) + torch.exp((-b + min_val) / gamma))) + min_val
    
    cell_mins = pos_tensor
    cell_maxs = pos_tensor + size_tensor
    cell_centers = pos_tensor + size_tensor / 2.0

    # Calculate pairwise overlap boundaries using smooth functions
    left_overlap = smooth_max(cell_mins[:, 0].unsqueeze(1), cell_mins[:, 0].unsqueeze(0))
    right_overlap = smooth_min(cell_maxs[:, 0].unsqueeze(1), cell_maxs[:, 0].unsqueeze(0))
    bottom_overlap = smooth_max(cell_mins[:, 1].unsqueeze(1), cell_mins[:, 1].unsqueeze(0))
    top_overlap = smooth_min(cell_maxs[:, 1].unsqueeze(1), cell_maxs[:, 1].unsqueeze(0))

    # Calculate smooth overlap width and height
    overlap_width = F.relu(right_overlap - left_overlap)
    overlap_height = F.relu(top_overlap - bottom_overlap)
    
    # The magnitude of the force is proportional to the smooth overlap area
    force_magnitude = overlap_width * overlap_height

    # Calculate the direction of the force (from center to center)
    cell_centers_i = cell_centers.unsqueeze(1)
    cell_centers_j = cell_centers.unsqueeze(0)
    direction = cell_centers_j - cell_centers_i
    direction_norm = torch.norm(direction, dim=-1, keepdim=True) + 1e-12
    normalized_direction = direction / direction_norm

    # Combine magnitude and direction
    force_vectors = force_magnitude.unsqueeze(-1) * normalized_direction

    # Mask out self-interaction
    self_mask = torch.eye(pos_tensor.size(0), dtype=torch.bool, device=pos_tensor.device)
    force_vectors = force_vectors.masked_fill(self_mask.unsqueeze(-1), 0.0)

    # Sum forces: force received by j from all i, and force applied by i to all j
    force_received = torch.sum(force_vectors, dim=0)
    force_applied = -torch.sum(force_vectors, dim=1)

    total_force = force_received + force_applied
    return total_force


def get_overlap_potential(size_tensor, pos_tensor, gamma=0.01):
    """
    Calculates a differentiable overlap potential based on smooth overlap area,
    consistent with get_overlap_diff.
    """
    def smooth_max(a, b):
        max_val = torch.max(a, b)
        return gamma * torch.log(torch.exp((a - max_val) / gamma) + torch.exp((b - max_val) / gamma)) + max_val

    def smooth_min(a, b):
        min_val = torch.min(a, b)
        return -gamma * (torch.log(torch.exp((-a + min_val) / gamma) + torch.exp((-b + min_val) / gamma))) + min_val

    cell_mins = pos_tensor
    cell_maxs = pos_tensor + size_tensor

    # Calculate pairwise overlap boundaries using smooth functions
    left_overlap = smooth_max(cell_mins[:, 0].unsqueeze(1), cell_mins[:, 0].unsqueeze(0))
    right_overlap = smooth_min(cell_maxs[:, 0].unsqueeze(1), cell_maxs[:, 0].unsqueeze(0))
    bottom_overlap = smooth_max(cell_mins[:, 1].unsqueeze(1), cell_mins[:, 1].unsqueeze(0))
    top_overlap = smooth_min(cell_maxs[:, 1].unsqueeze(1), cell_maxs[:, 1].unsqueeze(0))

    # Calculate smooth overlap width and height, ensuring non-negativity
    overlap_width = F.relu(right_overlap - left_overlap)
    overlap_height = F.relu(top_overlap - bottom_overlap)
    
    # The potential is the smooth overlap area
    overlap_area = overlap_width * overlap_height

    # Mask out self-overlap
    self_mask = torch.eye(size_tensor.size(0), dtype=torch.bool, device=size_tensor.device)
    overlap_area = overlap_area.masked_fill(self_mask, 0.0)

    # Total cost is half the sum of the symmetric matrix
    overlap_cost = torch.sum(overlap_area) / 2.0
    # Per-cell cost is the sum of its overlaps with all other cells
    overlap_list = torch.sum(overlap_area, dim=1)

    return overlap_cost, overlap_list
    
    
def get_pairwise_dist(pos_tensor, size_tensor):
    center = pos_tensor + size_tensor / 2.0             # (B, N, 2)
    diff = center.unsqueeze(2) - center.unsqueeze(1)    # (B, N, 2)

    dist = torch.norm(diff, dim=-1)                     # (B, N, N)
    return dist

def get_density_potential(pos_tensor, size_tensor, grid_size=10, area_max_size=1.0, target_density=0.8):
    """
    Calculates a potential-based density cost using a memory-efficient scatter-add approach.

    Args:
        pos_tensor (torch.Tensor): The bottom-left positions of the macros. Shape: (N, 2)
        size_tensor (torch.Tensor): The sizes of the macros. Shape: (N, 2)
        grid_size (int): The number of bins in each dimension of the grid.
        area_max_size (float): The maximum coordinate value for the placement area.
        target_density (float): The ideal density for a bin. Bins with density above this will be penalized.

    Returns:
        torch.Tensor: A scalar value representing the total density cost.
    """
    num_macros = pos_tensor.shape[0]
    device = pos_tensor.device

    # 1. Define the grid
    bin_size = area_max_size / grid_size
    bin_area = bin_size * bin_size
    
    # 2. Find the range of bins each macro overlaps with
    start_bins = torch.floor(pos_tensor / bin_size).long()
    end_bins = torch.floor((pos_tensor + size_tensor) / bin_size).long()

    # 3. Create lists of overlapping bin indices and areas for all macros
    all_overlap_areas = []
    all_bin_indices = []
    
    # This loop is over macros, but the inner operations are vectorized and efficient.
    # This approach is more memory-friendly than a fully vectorized one for sparse overlaps.
    for i in range(num_macros):
        macro_min = pos_tensor[i]
        macro_max = pos_tensor[i] + size_tensor[i]

        # Generate coordinates of all potentially overlapping bins for the current macro
        bin_x_coords = torch.arange(start_bins[i, 0], end_bins[i, 0] + 1, device=device)
        bin_y_coords = torch.arange(start_bins[i, 1], end_bins[i, 1] + 1, device=device)

        # Skip macros that don't fall into any bins
        if len(bin_x_coords) == 0 or len(bin_y_coords) == 0:
            continue

        grid_x, grid_y = torch.meshgrid(bin_x_coords, bin_y_coords, indexing='xy')
        
        # Clamp coordinates to be within the grid boundaries
        overlapping_bin_coords = torch.stack([
            torch.clamp(grid_x.flatten(), 0, grid_size - 1), 
            torch.clamp(grid_y.flatten(), 0, grid_size - 1)
        ], dim=1)

        # Calculate overlap area for this macro with its K overlapping bins
        bin_mins = overlapping_bin_coords.float() * bin_size
        bin_maxs = bin_mins + bin_size

        overlap_min = torch.max(macro_min.unsqueeze(0), bin_mins)
        overlap_max = torch.min(macro_max.unsqueeze(0), bin_maxs)
        
        overlap_size = torch.clamp(overlap_max - overlap_min, min=0.0)
        overlap_area = overlap_size[:, 0] * overlap_size[:, 1]

        # Get the flat indices for scattering
        flat_bin_indices = overlapping_bin_coords[:, 0] * grid_size + overlapping_bin_coords[:, 1]

        all_overlap_areas.append(overlap_area)
        all_bin_indices.append(flat_bin_indices)

    # 4. Concatenate lists and use scatter_add_ to sum areas into the density grid
    if not all_overlap_areas:
        return torch.tensor(0.0, device=device)

    src = torch.cat(all_overlap_areas)
    index = torch.cat(all_bin_indices)
    
    density_grid_flat = torch.zeros(grid_size * grid_size, device=device)
    density_grid_flat.scatter_add_(0, index, src)

    # 5. Normalize and calculate the final cost
    normalized_density = density_grid_flat / bin_area
    density_cost = torch.sum(torch.square(F.relu(normalized_density - target_density)))
    
    return density_cost
