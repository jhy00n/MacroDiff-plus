# Physics-Guided Geometric Diffusion for Macro Placement Generation

Official repository for the IJCAI 2026 paper **"Physics-Guided Geometric Diffusion for Macro Placement Generation".**

## Structure

```
.
├── models/              # Model architectures
│   ├── model.py        # MacroPlacer, TransPlacer, GraphPlacer
│   ├── diffuser.py     # MacroDiff diffusion model
│   ├── graph.py        # Graph processing modules
│   └── trans.py        # Transformer components
├── utils/              # Utility functions
│   ├── normalize.py    # Data normalization
│   ├── score.py        # HPWL and overlap metrics
│   ├── plot.py         # Visualization
│   ├── graph.py        # Benchmark parsing / dataset construction
│   └── write.py        # Benchmark file generation
├── dataset/            # Dataset directory
├── checkpoint/         # Model checkpoints
├── benchmarks/         # Circuit benchmarks (NOT included — see below)
├── map/                # Design mappings (cell-name ↔ tensor-index)
├── train.py            # Training script
├── test.py             # Inference and evaluation
└── Dockerfile          # Environment setup
```

## Environment Setup

### Using Docker (Recommended)

```bash
docker build -t macrodiff .
docker run --gpus all -it -v $(pwd):/workspace macrodiff
```

## Benchmarks

**The benchmarks are not distributed with this repository** (they are third-party
data and exceed GitHub's file size limits). `benchmarks/` is listed in `.gitignore`;
you must download them yourself before running anything.

Two benchmark sets are required:


| Set                         | Used for                                                      |
| --------------------------- | ------------------------------------------------------------- |
| **MMS** (Modern Mixed-Size) | nets / placements / row info; macros are movable              |
| **ISPD 2005**               | identifying which cells are macros (fixed `terminal` entries) |


Expected layout after download:

```
benchmarks/
├── mms/
│   ├── adaptec1/  adaptec1.nodes  adaptec1.nets  adaptec1.pl  adaptec1.scl  adaptec1.aux  ...
│   ├── adaptec2/  ...
│   └── bigblue4/  ...
└── ispd2005/
    ├── adaptec1/  adaptec1.nodes  ...
    └── ...
```

Designs used in the paper: `adaptec1-4`, `bigblue1-4`.

If you use these benchmarks, please cite the original works:

- J. Z. Yan, N. Viswanathan, and C. Chu, "Handling Complexities in Modern
Large-Scale Mixed-Size Placement," *DAC*, pp. 436–441, 2009. (MMS)
- G.-J. Nam et al., "The ISPD2005 Placement Contest and Benchmark Suite,"
*ISPD*, pp. 216–220, 2005. (ISPD 2005)

## Dataset

Pre-built evaluation tensors are provided:

- `dataset/test.pt` — test data
- `dataset/test_clustered.pt` — clustered test data (standard cells grouped into clusters)

Training data is not included due to size constraints; it is built from the
benchmarks above with `utils/graph.py`, which parses the `.nodes` / `.nets` /
`.pl` / `.scl` files into `HeteroData` graphs.

## Training

```bash
python train.py --data_path ./dataset --batch_size 8 --num_epochs 1000 --lr 1e-4
```

Key arguments:

- `--timesteps`: Diffusion timesteps (default: 200)
- `--model`: Model type — `full` | `trans` | `graph` (default: `full`)
- `--alpha`, `--beta`: Noise weights
- `--guide_lr`: Guidance learning rate
- `--max_iterations`: Max guidance iterations
- `--no_wandb`: Disable Weights &amp; Biases logging (otherwise a W&amp;B login is required)

## Inference

```bash
python test.py --checkpoint ./checkpoint/checkpoint.ckpt --result_path ./results --cluster
```

Key arguments:

- `--timesteps`: Diffusion timesteps (default: 200)
- `--cluster`: Sample with clustered standard cells. This also selects the
design map used to write the output `.pl` files — `map/test_clustered` with
`--cluster`, `map/test` without it — so the two never have to be kept in sync
by hand.
- `--map_path`: Directory holding the two maps (default: `./map`). Override
only if you keep them elsewhere; the `test` / `test_clustered` subdirectory is
still chosen by `--cluster`.
- `--guide_lr`: Guidance strength. If omitted, a per-design value (0.005–0.05) is used.
- `--threshold`: Convergence threshold. If omitted, a per-design value (0.05–0.5) is used.
- `--max_iterations`: Max guidance iterations. If omitted, a per-design value (300–700) is used.

The three guidance arguments fall back to the per-design defaults in
`param_dict` (`test.py`) only when not passed explicitly; any value you pass on
the command line is applied to every design.

## Outputs

- Placement visualizations (PNG, GIF)
- Benchmark files (MMS format `.pl` / `.aux`), written next to the reference
benchmark so they can be scored with the official MMS scripts

## License

Source code in this repository is released under the MIT License (see `LICENSE`).
The ISPD 2005 and MMS benchmarks are not covered by this license and remain
subject to the terms of their respective owners.

## Citation

```bibtex
@inproceedings{macrodiffplus2026,
  title     = {Physics-Guided Geometric Diffusion for Macro Placement Generation},
  author    = {Yoon, Jongho and Jeon, Jinsung and Kang, Seokhyeong},
  booktitle = {Proceedings of the Thirty-Fifth International Joint Conference on
               Artificial Intelligence (IJCAI)},
  year      = {2026}
}
```

