import os

os.environ['MPLCONFIGDIR'] = './.config'
os.environ['HF_HOME'] = './.cache/'
os.environ['XDG_CACHE_HOME'] = './.cache/'

import time
import random
import argparse
import logging
import wandb
from tqdm import tqdm

import torch
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from diffusers.optimization import get_cosine_schedule_with_warmup

from models.model import MacroPlacer, TransPlacer, GraphPlacer
from models.diffuser import MacroDiff
from utils.normalize import normalize, unnormalize, log_normalize
from utils.score import (
    get_overlap_tensor, 
    get_hpwl_tensor, 
)
from utils.plot import plot_design, plot_gif_design
from utils.write import write_bench_files

torch.set_printoptions(profile="full")

design_list = ['adaptec1', 'adaptec2', 'adaptec3', 'adaptec4', 'bigblue1', 'bigblue2', 'bigblue3', 'bigblue4', ]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--device', type=int, default=0, 
                    help="CUDA device index (default: 0)")
parser.add_argument('--data_path', type=str, default='./dataset', 
                    help="Path to the dataset (default: ./dataset)")
parser.add_argument('-r', '--result_path', type=str, default='./result', 
                    help="Path to the result (default: ./result)")
parser.add_argument('--bench_path', type=str, default='./benchmarks/mms', 
                    help="Path to the benchmarks (default: ./benchmarks/mms)")
parser.add_argument('--map_path', type=str, default='./map/test', 
                    help="Path to the map (default: ./map/test)")
parser.add_argument('-b', '--batch_size', type=int, default=8, 
                    help="size of batch (default: 8)")
parser.add_argument('-t', '--timesteps', type=int, default=200, 
                    help="timesteps (default: 200)")
parser.add_argument('-m', '--model', type=str, default='full', 
                    help="model type (default: full)")
parser.add_argument('-n', '--noise', type=str, default='full', 
                    help="model type (default: full)")
parser.add_argument('--start_epoch', type=int, default=0, 
                    help="start epoch (default: 0)")
parser.add_argument('--num_epochs', type=int, default=1000, 
                    help="train epochs (default: 1000)")
parser.add_argument('--lr', type=float, default=1e-4, 
                    help="learning rate (default: 1e-4)")
parser.add_argument('--lr_warmup_steps', type=int, default=50000,
                    help="Number of steps for the warmup phase (default: 50000)")
parser.add_argument('--dropout', type=float, default=0., 
                    help="dropout rate (default: 0.)")
parser.add_argument('--weight_decay', type=float, default=0., 
                    help="weight decay (default: 0.)")
parser.add_argument('--num_heads', type=int, default=4, 
                    help="number of heads (default: 4)")
parser.add_argument('--alpha', type=float, default=1.0, 
                    help="cell noise weight (default: 1.0)")
parser.add_argument('--beta', type=float, default=1.0, 
                    help="net noise weight (default: 1.0)")
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training.')
parser.add_argument('--max_iterations', type=int, default=100, 
                    help="max iterations (default: 100)")
parser.add_argument('--guide_lr', type=float, default=0.01, 
                    help="learning rate of gradient-based guidance (default: 0.01)")
parser.add_argument('--weight_lr', type=float, default=5e-4, 
                    help="learning rate of guidance loss weight (default: 5e-4)")
parser.add_argument('--threshold', type=float, default=1e-4, 
                    help="threshold (default: 1e-4)")
parser.add_argument('--guide_x0', type=bool, default=True,
                    help='guide x0 (default: True)')
parser.add_argument('--no_wandb', action='store_true',
                    help='disable Weights & Biases logging')
                    

class CircuitDataset(Dataset):
    def __init__(self, dataset_path, train=True):
        init_start = time.time()
        super().__init__()
        self.dataset_path = dataset_path
        self.data_list = []

        cache_dir = os.path.join(dataset_path, 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache_name = f"{'train' if train else 'test'}_processed.pt"
        cache_file = os.path.join(cache_dir, cache_name)
        if os.path.exists(cache_file):
            logger.info(f"Loading preprocessed data from cache: {cache_file}")
            cache_start = time.time()
            self.data_list = torch.load(cache_file)
            cache_time = time.time() - cache_start
            logger.info(f"Cache loading took {cache_time:.2f}s")
            return

        if train:
            design_file = dataset_path + '/train.pt'
        else:
            design_file = dataset_path + '/test.pt'
        if not os.path.exists(design_file):
            raise FileNotFoundError(f'Data file {design_file} does not exist.')

        load_start = time.time()
        data = torch.load(design_file)
        load_time = time.time() - load_start
        logger.info(f"Data file loading took {load_time:.2f}s")

        max_start = time.time()
        max_macro = 0
        max_net = 0
        max_io = 0 

        total_samples = sum(len(data[design]) for design in design_list)

        for design in design_list:
            for sample in data[design]:
                max_macro = max(max_macro,sample['num_macro'])
                max_io = max(max_io,sample['num_io'])
                max_net = max(max_net,sample['net']['degree'].shape[0])
        
        max_time = time.time() - max_start
        logger.info(f"Max value calculation took {max_time:.2f}s")

        preprocess_start = time.time()
        num_samples = 0
        
        pbar = tqdm(total=total_samples, desc="Preprocessing samples")
        for design in design_list:
            for sample in data[design]:
                preprocess_sample_start = time.time()
                
                # 3.1 Position normalization
                pos_start = time.time()
                sample['cell']['pos'] = normalize(sample['cell']['pos'], sample['max_size'])
                sample['cell']['size'] = normalize(sample['cell']['size'], sample['max_size']) + 1.0
                pos_time = time.time() - pos_start
                
                # 3.2 Graph structure processing
                graph_start = time.time()
                sample['leng'] = len(sample['cell']['pos'])
                sample['cell']['degree_size'] = len(sample['net']['degree'])
                sample['net']['degree'] = log_normalize(sample['net']['degree'])
                sample['num_edge'] = len(sample['cell', 'out', 'net'].edge_attr)
                graph_time = time.time() - graph_start
                
                # 3.3 Edge attribute processing
                edge_start = time.time()
                sample['cell', 'out', 'net']['offset'] = sample['cell', 'out', 'net'].edge_attr.clone()
                sample['cell', 'out', 'net'].edge_attr = normalize(sample['cell', 'out', 'net'].edge_attr, sample['max_size'])
                sample['net', 'in', 'cell']['offset'] = sample['net', 'in', 'cell'].edge_attr.clone()
                sample['net', 'in', 'cell'].edge_attr = normalize(sample['net', 'in', 'cell'].edge_attr, sample['max_size'])
                edge_time = time.time() - edge_start
                
                # 3.4 Final feature assignment
                feat_start = time.time()
                sample['cell'].x = sample['cell']['pos'].clone()
                sample['net'].x = sample['net']['degree'].clone()
                sample['num_net'] = sample['net']['degree'].shape[0]
                sample['max_macro'] = max_macro
                sample['max_net'] = max_net
                sample['max_io'] = max_io
                feat_time = time.time() - feat_start
                
                sample_time = time.time() - preprocess_sample_start
                if num_samples == 0:
                    logger.info(f"First sample preprocessing breakdown:")
                    logger.info(f"  - Position normalization: {pos_time:.4f}s")
                    logger.info(f"  - Graph structure: {graph_time:.4f}s")
                    logger.info(f"  - Edge attributes: {edge_time:.4f}s")
                    logger.info(f"  - Feature assignment: {feat_time:.4f}s")
                    logger.info(f"  - Total sample time: {sample_time:.4f}s")
                
                self.data_list.append(sample)
                num_samples += 1
                pbar.update(1)
        pbar.close()

        logger.info(f"Saving preprocessed data to cache: {cache_file}")
        cache_save_start = time.time()
        torch.save(self.data_list, cache_file)
        cache_save_time = time.time() - cache_save_start
        logger.info(f"Cache saving took {cache_save_time:.2f}s")
        
        preprocess_time = time.time() - preprocess_start
        total_time = time.time() - init_start
        
        logger.info(f"Dataset initialization summary:")
        logger.info(f"  - File loading: {load_time:.2f}s ({100*load_time/total_time:.1f}%)")
        logger.info(f"  - Max calculation: {max_time:.2f}s ({100*max_time/total_time:.1f}%)")
        logger.info(f"  - Preprocessing: {preprocess_time:.2f}s ({100*preprocess_time/total_time:.1f}%)")
        logger.info(f"  - Average time per sample: {preprocess_time/num_samples:.4f}s")
        logger.info(f"  - Total samples: {num_samples}")
        logger.info(f"  - Total time: {total_time:.2f}s")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

        
def main():
    args = parser.parse_args()
    args.device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.device)

    if not args.no_wandb:
        wandb.init(project='IJCAI2026', config=args)

    experiment_dir = os.path.join(args.result_path, f"model_{args.model}_noise_{args.noise}")
    gif_dir = os.path.join(experiment_dir, 'gif')
    sample_dir = os.path.join(experiment_dir, 'sample')
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoint')
    place_dir = os.path.join(experiment_dir, 'placement')

    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(gif_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(place_dir, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    logger.info("Loading datasets...")
    train_dataset = CircuitDataset(args.data_path, train=True)
    test_dataset = CircuitDataset(args.data_path, train=False)

    num_workers = min(8, os.cpu_count())
    logger.info(f"Using {num_workers} workers for data loading")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

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
    diffusion = MacroDiff(model, noise=args.noise, timesteps=args.timesteps, alpha=alpha, beta=beta, max_iterations=args.max_iterations, guide_lr=args.guide_lr, weight_lr=args.weight_lr, threshold=args.threshold)

    model.to(args.device)
    diffusion.to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer, 
        num_warmup_steps=args.lr_warmup_steps, 
        num_training_steps=(len(train_loader) * args.num_epochs)
    )

    if args.pretrained or args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=args.device)
            if 'epoch' in checkpoint:
                args.start_epoch = checkpoint['epoch'] + 1
                print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
                print(f"=> resuming from epoch {args.start_epoch}")
            else:
                import re
                filename = os.path.basename(args.resume)
                epoch_match = re.search(r'epoch(\d+)', filename)
                if epoch_match:
                    saved_epoch = int(epoch_match.group(1))
                    args.start_epoch = saved_epoch + 1
                    print(f"=> loaded checkpoint '{args.resume}' (epoch {saved_epoch})")
                    print(f"=> resuming from epoch {args.start_epoch}")
                else:
                    print(f"=> loaded checkpoint '{args.resume}' (epoch info not found)")
                    print(f"=> starting from epoch {args.start_epoch}")
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print(f"=> no checkpoint found at '{args.resume}'")
    
    hpwl_dict = {}
    overlap_dict = {}

    ##### Get HPWL and overlap for baseline design #####
    for i, data in enumerate(test_loader):
        data = data.to(args.device)
        size = data['cell']['size'].clone()
        pos = data['cell']['pos'].clone()
        num_io = data['num_io']
        num_macro = data['num_macro']
        size_test = unnormalize(size-1.0, data['max_size'])
        pos_test = unnormalize(pos, data['max_size'])

        write_bench_files(design_list[i], pos_test, args.bench_path, f'{experiment_dir}/result', args.map_path, sign='base')
        hpwl, _ = get_hpwl_tensor(data, pos_test)
        overlap, _ = get_overlap_tensor(size_test, pos_test) 

        hpwl_dict[design_list[i]] = hpwl
        overlap_dict[design_list[i]] = overlap

    if not args.no_wandb:
        log_data = {'epoch': 0}
        for d in design_list:
            log_data[f'{d}/hpwl'] = hpwl_dict[d]
            log_data[f'{d}/overlap'] = overlap_dict[d]
        wandb.log(log_data)

    logger.info("Start training...")
    for e in range(args.start_epoch, args.num_epochs):
        diffusion.model.train()
        total_loss = 0.0
        epoch_start_time = time.time()
        data_loading_time = 0
        forward_time = 0
        backward_time = 0
        optimizer_time = 0
        total_iter_gap = 0

        logger.info(f"Starting epoch {e+1}")
        for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {e+1}")):
            if i > 0:
                iter_gap = time.time() - iter_end_time
                total_iter_gap += iter_gap
                if iter_gap > 1.0:
                    logger.info(f"Large gap between iterations {i-1} and {i}: {iter_gap:.2f}s")

            iter_start_time = time.time()

            data_load_start = time.time()
            data = data.to(args.device)
            data_loading_time += time.time() - data_load_start

            forward_start = time.time()
            optimizer.zero_grad()
            loss = diffusion.forward(data)
            total_loss += loss.item()
            forward_time += time.time() - forward_start

            backward_start = time.time()
            loss.backward()
            backward_time += time.time() - backward_start

            optimizer_start = time.time()
            optimizer.step()
            lr_scheduler.step()
            optimizer_time += time.time() - optimizer_start

            iter_end_time = time.time()

            if i % 10 == 0:
                current_time = time.time() - epoch_start_time
                logger.info(f"Iteration {i}: Loss={loss.item():.4f}, "
                          f"Time breakdown - Data: {data_loading_time/(i+1):.4f}s, "
                          f"Forward: {forward_time/(i+1):.4f}s, "
                          f"Backward: {backward_time/(i+1):.4f}s, "
                          f"Optimizer: {optimizer_time/(i+1):.4f}s, "
                          f"Iter gaps: {total_iter_gap/(i+1):.4f}s")

        epoch_time = time.time() - epoch_start_time
        avg_data_loading = data_loading_time / (i + 1)
        avg_forward = forward_time / (i + 1)
        avg_backward = backward_time / (i + 1)
        avg_optimizer = optimizer_time / (i + 1)
        avg_iter_gap = total_iter_gap / i if i > 0 else 0

        logger.info(f"Epoch {e+1}/{args.num_epochs}")
        logger.info(f"Total time: {epoch_time:.2f}s")
        logger.info(f"Average times per batch:")
        logger.info(f"  - Data loading: {avg_data_loading:.4f}s ({100*avg_data_loading/epoch_time:.1f}%)")
        logger.info(f"  - Forward pass: {avg_forward:.4f}s ({100*avg_forward/epoch_time:.1f}%)")
        logger.info(f"  - Backward pass: {avg_backward:.4f}s ({100*avg_backward/epoch_time:.1f}%)")
        logger.info(f"  - Optimizer step: {avg_optimizer:.4f}s ({100*avg_optimizer/epoch_time:.1f}%)")
        logger.info(f"  - Average gap between iterations: {avg_iter_gap:.4f}s ({100*avg_iter_gap/epoch_time:.1f}%)")
        logger.info(f"  - Loss: {total_loss:.4f}")
        logger.info(f"  - Unaccounted time: {epoch_time - (avg_data_loading + avg_forward + avg_backward + avg_optimizer + avg_iter_gap)*(i+1):.2f}s")

        if (e + 1) % 50 == 0:
            logger.info(f"Validation...")
            model.eval()
            hpwl_dict = {}
            overlap_dict = {}
            for i, data in enumerate(test_loader):
                design = design_list[i]

                data = data.to(args.device)
                num_io = data['num_io']
                num_macro = data['num_macro']
                size = data['cell']['size'].clone()
                pos = data['cell']['pos'].clone()
                size_test = unnormalize(size-1.0, data['max_size'])
                pos_test = unnormalize(pos, data['max_size'])

                io, x_0 = pos.split([num_io, num_macro])
                sample, sample_list = diffusion.guided_valid(data, guide_x0=args.guide_x0)

                plot_gif_design(sample_list, size[-num_macro:], save_dir=gif_dir, design=f'{design}_{e+1}')

                pos_0 = torch.cat([io, sample.squeeze()], dim=0)
                pos_final = unnormalize(pos_0, data['max_size'])
                pos_final = pos_final.clamp(min=torch.zeros_like(pos_final), max=data['max_size']-size_test)
                plot_design(pos_final, size_test, data['max_size'], save_dir=sample_dir, design=design_list[i], time=(e+1))

                write_bench_files(design_list[i], pos_final, args.bench_path, f'{experiment_dir}/result', args.map_path, sign=f'{args.model}_{args.noise}_{e+1}')
                hpwl, _ = get_hpwl_tensor(data, pos_final)
                overlap, _ = get_overlap_tensor(size_test, pos_final) 

                hpwl_dict[design_list[i]] = hpwl
                overlap_dict[design_list[i]] = overlap
            
                logger.info(f"HPWL for {design}: {hpwl:.4f}, Overlap: {overlap:.4f}")

            logger.info(f"Saving model to {checkpoint_dir}")
            torch.save({
                'epoch': e + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(checkpoint_dir, f'epoch{e+1}.ckpt'))

            if not args.no_wandb:
                log_data = {'epoch': e+1, 'loss': total_loss}
                for d in design_list:
                    log_data[f'{d}/hpwl'] = hpwl_dict[d]
                    log_data[f'{d}/overlap'] = overlap_dict[d]
                wandb.log(log_data)

        else:
            if not args.no_wandb:
                wandb.log({'epoch': e+1, 'loss': total_loss})
    

if __name__ == '__main__':
    main()
    