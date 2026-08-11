import os
import time
import random
import argparse
import logging
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from models.model import MacroPlacer, TransPlacer, GraphPlacer
from models.diffuser import MacroDiff
from utils.normalize import normalize, unnormalize, log_normalize
from utils.write import write_bench_files
from utils.plot import plot_design, plot_gif_design

design_list = ['adaptec1', 'adaptec2', 'adaptec3', 'adaptec4', 'bigblue1', 'bigblue2', 'bigblue3', 'bigblue4', ]

# Per-design guidance defaults, used only for arguments the user did not pass explicitly.
param_dict = {
    'adaptec1': {'iters': 300, 'lr': 0.05, 'threshold': 0.5},
    'adaptec2': {'iters': 700, 'lr': 0.05, 'threshold': 0.5},
    'adaptec3': {'iters': 700, 'lr': 0.05, 'threshold': 0.1},
    'adaptec4': {'iters': 700, 'lr': 0.05, 'threshold': 0.1},
    'bigblue1': {'iters': 300, 'lr': 0.05, 'threshold': 0.1},
    'bigblue2': {'iters': 700, 'lr': 0.01, 'threshold': 0.05},
    'bigblue3': {'iters': 700, 'lr': 0.005, 'threshold': 0.1},
    'bigblue4': {'iters': 500, 'lr': 0.05, 'threshold': 0.05},
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--device', type=int, default=0, 
                    help="CUDA device index (default: 0)")
parser.add_argument('-c', '--checkpoint', type=str, default='./checkpoint/checkpoint.ckpt', 
                    help="Path to the model checkpoint (default: ./checkpoint/checkpoint.ckpt)")
parser.add_argument('--data_path', type=str, default='./dataset', 
                    help="Path to the dataset (default: ./dataset)")
parser.add_argument('--bench_path', type=str, default='./benchmarks/mms', 
                    help="Path to the benchmarks (default: ./benchmarks/mms)")
parser.add_argument('--map_path', type=str, default='./map',
                    help="Path to the map directory; the 'test' / 'test_clustered' "
                         "subdirectory is selected by --cluster (default: ./map)")
parser.add_argument('-r', '--result_path', type=str, default='./sample_v2', 
                    help="Path to the result (default: ./sample)")
parser.add_argument('-t', '--timesteps', type=int, default=200, 
                    help="timesteps (default: 200)")
parser.add_argument('-m', '--model', type=str, default='full', 
                    help="model type (default: full)")
parser.add_argument('-n', '--noise', type=str, default='full', 
                    help="model type (default: full)")
parser.add_argument('--dropout', type=float, default=0., 
                    help="dropout rate (default: 0.)")
parser.add_argument('--num_heads', type=int, default=4, 
                    help="number of heads (default: 4)")
parser.add_argument('--alpha', type=float, default=1., 
                    help="net noise weight (default: 1.)")
parser.add_argument('--beta', type=float, default=1., 
                    help="cell noise weight (default: 1.)")
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training.')
parser.add_argument('--max_iterations', type=int, default=None,
                    help="max guidance iterations (default: per-design value, 300-700)")
parser.add_argument('--guide_lr', type=float, default=None,
                    help="learning rate of gradient-based guidance (default: per-design value, 0.005-0.05)")
parser.add_argument('--threshold', type=float, default=None,
                    help="convergence threshold (default: per-design value, 0.05-0.5)")
parser.add_argument('--guide_x0', type=bool, default=True,
                    help='guide x0 (default: True)')
parser.add_argument('--cluster', action='store_true',
                    help='sampling includes cluster')


class CircuitDataset(Dataset):
    def __init__(self, dataset_path, train=False, cluster=False):
        super().__init__()
        self.dataset_path = dataset_path
        self.data_list = []

        if train:
            design_file =  dataset_path + '/train.pt'
        else:
            if cluster:
                design_file =  dataset_path + '/test_clustered.pt'
            else:
                design_file =  dataset_path + '/test.pt'
        if not os.path.exists(design_file):
            raise FileNotFoundError(f'Data file {design_file} does not exist.')
        data = torch.load(design_file)
 
        for design in design_list:
            for sample in data[design]:
                sample['cell']['pos'] = normalize(sample['cell']['pos'], sample['max_size'])
                sample['cell']['size'] = normalize(sample['cell']['size'], sample['max_size']) + 1.0
                sample['leng'] = len(sample['cell']['pos'])
                sample['cell']['degree_size'] = len(sample['net']['degree'])
                sample['net']['degree'] = log_normalize(sample['net']['degree'])
                sample['num_edge'] = len(sample['cell', 'out', 'net'].edge_attr)
                sample['cell', 'out', 'net']['offset'] = sample['cell', 'out', 'net'].edge_attr.clone()
                sample['cell', 'out', 'net'].edge_attr = normalize(sample['cell', 'out', 'net'].edge_attr, sample['max_size'])
                sample['net', 'in', 'cell']['offset'] = sample['net', 'in', 'cell'].edge_attr.clone()
                sample['net', 'in', 'cell'].edge_attr = normalize(sample['net', 'in', 'cell'].edge_attr, sample['max_size'])
                sample['cell'].x = sample['cell']['pos'].clone()
                sample['net'].x = sample['net']['degree'].clone()
                sample['num_net'] = sample['net']['degree'].shape[0]
                self.data_list.append(sample)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
    

def main():
    args = parser.parse_args()
    args.device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)

    exp_dir = os.path.join(args.result_path, f"model_{args.model}_noise_{args.noise}")
    gif_dir = os.path.join(exp_dir, 'gif')
    sample_dir = os.path.join(exp_dir, 'sample')
    place_dir = os.path.join(exp_dir, 'placement')
    
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(gif_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(place_dir, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    # The map must match the index space of the positions handed to write_bench_files:
    #   ./map/test           -> [IO ..., macro ...]      i.e. the full pos tensor
    #   ./map/test_clustered -> [macro ..., cluster ...] i.e. pos[num_io:]
    map_path = os.path.join(args.map_path, 'test_clustered' if args.cluster else 'test')

    logger.info("Loading datasets...")
    test_dataset = CircuitDataset(args.data_path, train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)
    if args.cluster:
        test_cluster_dataset = CircuitDataset(args.data_path, train=False, cluster=True)
        test_cluster_loader = DataLoader(test_cluster_dataset, batch_size=1, shuffle=False, pin_memory=True)
    else:
        test_cluster_dataset = None
        test_cluster_loader = [None] * len(test_loader)

    logger.info("Initializing model...")
    #########################
    in_cell_channels = 5
    in_net_channels = 3
    hidden_channels = 64
    edge_dim = 2
    num_heads = args.num_heads
    alpha = args.alpha
    beta = args.beta
    dropout = args.dropout
    #########################

    if args.model == 'full':
        model = MacroPlacer(in_cell_channels, in_net_channels, hidden_channels, edge_dim, num_heads=num_heads, dropout=dropout)
    elif args.model == 'trans':
        model = TransPlacer(in_cell_channels, hidden_channels, edge_dim, num_heads=num_heads, dropout=dropout)
        if args.noise == 'full':
            raise ValueError(f"Invalid model & noise match: {args.model} & {args.noise}") 
    elif args.model == 'graph':
        model = GraphPlacer(in_cell_channels, in_net_channels, hidden_channels, edge_dim, num_heads=num_heads, dropout=dropout)
    else:
        raise ValueError(f"Invalid model: {args.model}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    # max_iterations / guide_lr / threshold are set per design in the loop below.
    diffusion = MacroDiff(model, noise=args.noise, timesteps=args.timesteps, alpha=alpha, beta=beta)

    model.to(args.device)
    diffusion.to(args.device)

    model.eval()

    # Initialize timing statistics dictionary
    timing_stats = defaultdict(list)

    for i, (data, data_cluster) in enumerate(zip(test_loader, test_cluster_loader)):
        design = design_list[i]

        data = data.to(args.device)
        num_io = data['num_io']
        num_macro = data['num_macro']
        pos = data['cell']['pos'].clone()
        if args.cluster:
            data_cluster = data_cluster.to(args.device)
            num_cluster = data_cluster['num_cluster']
            size = data_cluster['cell']['size'].clone()
        else:
            num_cluster = 0
            size = data['cell']['size'].clone()

        io, x_0 = pos.split([num_io, num_macro])

        # An explicitly passed CLI value wins; otherwise fall back to the per-design default.
        diffusion.max_iterations = args.max_iterations if args.max_iterations is not None else param_dict[design]['iters']
        diffusion.guide_lr = args.guide_lr if args.guide_lr is not None else param_dict[design]['lr']
        diffusion.threshold = args.threshold if args.threshold is not None else param_dict[design]['threshold']
        logger.info(f"{design} - iters={diffusion.max_iterations}, guide_lr={diffusion.guide_lr}, threshold={diffusion.threshold}")

        for j in range(0, 10):
            # Measure sampling time
            start_time = time.time()
            sample, sample_list = diffusion.guided_valid(data, guide_x0=True, seed=j, w_hpwl=0.001, w_overlap=1.0, cluster=args.cluster, data_cluster=data_cluster)
            sampling_time = time.time() - start_time

            # Store timing for this design
            timing_stats[design].append(sampling_time)
            logger.info(f"{design} - Sample {j+1}/10: {sampling_time:.2f}s")
            # Conditional slicing to avoid empty tensor when num_cluster=0
            macro_size = size[-num_cluster-num_macro:-num_cluster] if num_cluster > 0 else size[num_io:]
            plot_gif_design(sample_list, macro_size, save_dir=gif_dir, design=f'{design}_{j}')


            pos_0 = torch.cat([io, sample], dim=0)
            pos_final = unnormalize(pos_0, data['max_size'])

            size_final = unnormalize(data['cell']['size'] - 1.0, data['max_size'])
            pos_final = pos_final.clamp(min=torch.zeros_like(pos_final), max=data['max_size']-size_final)
            plot_design(
                pos=pos_final,
                size=size_final,
                max_size=data['max_size'],
                save_dir=sample_dir,
                design=design_list[i],
                time=j,
            )

            pos_write = pos_final[num_io:] if args.cluster else pos_final
            write_bench_files(design_list[i], pos_write, args.bench_path, f'{exp_dir}/result', map_path, sign=f'{args.model}_{j}')

    # Print timing statistics summary
    logger.info("\n" + "="*60)
    logger.info("TIMING STATISTICS SUMMARY")
    logger.info("="*60)

    for design in design_list:
        if design in timing_stats:
            times = timing_stats[design]
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            logger.info(f"{design:12s} | Avg: {avg_time:7.2f}s | Min: {min_time:7.2f}s | Max: {max_time:7.2f}s | Total: {sum(times):7.2f}s")

    logger.info("="*60)


if __name__ == '__main__':
    main()
