# GPU-Arch 3DGS Experiments (`gsplat-fork`)

This repo is a course/project fork focused on improving `trainer_refactored.py` runtime and VRAM behavior for Gaussian Splatting experiments (cache policy, prefetch, frustum culling, and optimizer stride sweeps), with SLURM + Nsight Systems profiling support.

## What matters in this repo

- Main trainer: `examples/trainer_refactored.py`
- Benchmark entrypoint used by SLURM: `examples/benchmarks/custom_rubble.sh`
- SLURM job generator: `scripts/slurm/slurm_scheduler.py`
- Job template: `scripts/slurm/profile_gsplat_template_rivanna.slurm`
- Stats aggregation: `scripts/slurm/aggregate_model_stats.py`
- Nsight export helper: `scripts/slurm/nsys_export_sqlite.sh`
- Nsight table builder: `scripts/slurm/nsys_kernel_table.py`

## Environment setup

1. Install PyTorch first (CUDA build that matches your system).
2. From repo root:

```bash
pip install -e .
pip install -r examples/requirements.txt --no-build-isolation
```

## Data setup

The trainer expects COLMAP-style scene folders.

### Option A: Download MipNeRF360 benchmark scenes

```bash
cd examples
python datasets/download_dataset.py --dataset mipnerf360 --save_dir data
```

This creates scenes under `examples/data/360_v2/...`.

### Option B: Use your custom rubble dataset

Place your scene at `examples/data/rubble-colmap/` with at least:

- `images/`
- `sparse/0/` (or `sparse/`)

`trainer_refactored.py`/`custom_rubble.sh` are already wired for `data/rubble-colmap`.

## Quick local run (no SLURM)

```bash
cd examples
CUDA_VISIBLE_DEVICES=0 python -u trainer_refactored.py default \
  --disable_viewer \
  --data_dir data/rubble-colmap \
  --result_dir results/rubble_local \
  --cache_mode lru \
  --enable_frustum_culling \
  --optimizer_stride 1
```

## Running SLURM experiments

1. Edit sweep and fixed settings in `scripts/slurm/slurm_scheduler.py`:
   `fixed_params` (scene path, GPU/mem/time, script path), `sweep_params` (ablations), and `mode` (`zip` or `grid`).
2. Submit jobs:

```bash
cd scripts/slurm
python slurm_scheduler.py
```

3. Outputs are created in a timestamped folder under:

`scripts/slurm/outputs/<timestamp>_<prefix>/`

You’ll get:
- `slurm_scripts/`
- `slurm_out/`
- `slurm_err/`
- `configs/`
- `results/` (`.nsys-rep`, `.sqlite`, trainer outputs)

4. Aggregate trainer stats:

```bash
python aggregate_model_stats.py outputs/<timestamp>_<prefix>
```

This writes `stats.csv` in that output root.

## Nsight profiling post-processing

If `.sqlite` files are not already exported:

```bash
cd scripts/slurm
./nsys_export_sqlite.sh outputs/<timestamp>_<prefix>/results
```

Then build memory/utilization summary tables:

```bash
python nsys_kernel_table.py outputs/<timestamp>_<prefix>/results --output nsys_mem_table.csv
```

## Key `trainer_refactored.py` args

### Data/output

- `--data_dir`: scene path (`data/rubble-colmap`, `data/360_v2/garden`, ...)
- `--data_factor`: image downsample factor
- `--result_dir`: output root for ckpts/stats/renders/videos
- `--test_every`: train/val split cadence

### Schedule/checkpoints

- `--steps_scaler`: scales many step-based schedules together
- `--max_steps`: total training steps
- `--eval_steps`: evaluation checkpoints
- `--save_steps`: checkpoint save steps
- `--ply_steps`: ply export steps (if enabled)

### Runtime/VRAM tuning (main experiment knobs)

- `--optimizer_stride`: gradient accumulation stride before optimizer step
- `--cache_mode`: `none | lru | lfu | twoq | warm_all`
- `--enable_prefetch`: async GPU prefetch for upcoming batches
- `--prefetch_lookahead`: prefetch queue lookahead depth
- `--vram_thresh_gb`: GPU cache budget
- `--twoq_a1_ratio`: A1 queue fraction in 2Q mode
- `--enable_input_cache`: legacy toggle (enables cache when `cache_mode=none`)
- `--enable_frustum_culling`: pre-raster cull mask
- `--frustum_cull_interval`: how often masks are recomputed
- `--frustum_cull_radius_scale`, `--frustum_cull_margin_*`: culling aggressiveness

### Quality/feature toggles

- `--ssim_lambda`: L1 vs SSIM loss mix
- `--depth_loss`, `--depth_lambda`: depth supervision
- `--pose_opt`, `--app_opt`: camera/appearance optimization
- `--post_processing`: `bilateral_grid` or `ppisp` (single GPU only)

### Eval/render path

- `--ckpt`: eval-only mode on saved checkpoint(s)
- `--render_traj_path`: `interp | ellipse | spiral`
- `--disable_video`: skip trajectory video generation

## Important notes

- `custom_rubble.sh` runs train, then eval/render on produced checkpoints.
- `profile_gsplat_template_rivanna.slurm` assumes a Rivanna-style module workflow and an existing venv at `/scratch/rhm4nj/gpu_arch/envs/<gpu_type>-env`.
- If you move this repo, update absolute paths in `scripts/slurm/slurm_scheduler.py` and possibly the SLURM template.
- Default config preset is `default`; `mcmc` is also available via:

```bash
python trainer_refactored.py mcmc ...
```
