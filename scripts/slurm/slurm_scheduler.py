from pathlib import Path
import itertools
import subprocess
import sys
from datetime import datetime

# All code will be RUN on Rivanna (/scratch directory) - DO NOT mess with this file path!!!
project_dir = Path("/scratch/rhm4nj/gpu_arch/gsplat-fork")
# project_dir = Path("/bigtemp/rhm4nj/gpu_arch/project/gsplat-fork")
TEMPLATE_PATH = str(project_dir / "scripts/slurm/profile_gsplat_template_rivanna.slurm")

mode = "zip"   # "grid" = cartesian product, "zip" = pair by index (all lists must be same length)
prefix = "_QUICK_frustum_gid_postrefine_with_true_baseline"
MAX_JOBS = 8

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
base_dir = project_dir / "scripts/slurm/outputs" / Path(timestamp + prefix)

slurm_dir      = base_dir / "slurm_scripts"
slurm_out_dir  = base_dir / "slurm_out"
slurm_err_dir  = base_dir / "slurm_err"
config_dir     = base_dir / "configs"
results_dir    = base_dir / "results"

slurm_dir.mkdir(parents=True, exist_ok=False)
slurm_out_dir.mkdir(parents=True, exist_ok=False)
slurm_err_dir.mkdir(parents=True, exist_ok=False)
config_dir.mkdir(parents=True, exist_ok=False)
results_dir.mkdir(parents=True, exist_ok=False)

fixed_params = {
    "script_path": project_dir / "examples/benchmarks/custom_rubble.sh",
    "partition": "gpu",
    "gpu_num": 0,
    "log_out": slurm_out_dir / "%x-%j.out",
    "log_err": slurm_err_dir / "%x-%j.err",
    "trace": "cuda,osrt,nvtx",
    "sample": "cpu",
    "project_dir": project_dir,
    "gpus": 1,
    "cpus": 4,
    "mem": "64G",
    "time": "05:00:00",
    "force_overwrite": "true",
    "gpu_type": "a6000",
    "--optimizer-stride": 1,
    "scenes": "data/rubble-colmap",
}

# Quick three-way comparison:
# 1. true baseline: no cache, no prefetch, no culling
# 2. patched control: lru cache, no prefetch, no culling
# 3. patched culling: lru cache, no prefetch, culling after refine_stop_iter
sweep_params = {
    "--vram-thresh-gb": [10.0, 10.0, 10.0],
    "--cache-mode": ["none", "lru", "lru"],
    "--enable-prefetch": [False, False, False],
    "--enable-frustum-culling": [False, False, True],
}



# --- Helpers ---
profile_prefix = str(results_dir / "profile_gsplat")

def _abbrev(key: str) -> str:
    """Abbreviate a --kebab-case key: --enable-frustum-culling → efc"""
    return "".join(w[0] for w in key.lstrip("-").split("-"))

def make_profile_output(prefix: str, keys: list, combo: tuple) -> str:
    """Build: profile_gsplat_os-1_efc-True"""
    parts = [f"{_abbrev(k)}-{v}" for k, v in zip(keys, combo)]
    return "_".join([prefix] + parts)

def _format_arg(key: str, value) -> str:
    """
    Format a single --key/value pair for the shell command.
    Boolean flags:
      True  → --flag-name
      False → --no-flag-name
    """
    if isinstance(value, bool):
        return key if value else "--no-" + key.lstrip("-")
    return f"{key} {value}"

def build_job_params(fixed: dict, keys: list, combo: tuple) -> dict:
    """
    Build the job_params dict for one combination.
    """
    params = {k: v for k, v in fixed.items() if not str(k).startswith("--")}

    for k, v in zip(keys, combo):
        if not str(k).startswith("--"):
            params[k] = v

    args_parts = []
    for k, v in fixed.items():
        if str(k).startswith("--"):
            args_parts.append(_format_arg(k, v))

    for k, v in zip(keys, combo):
        if str(k).startswith("--"):
            args_parts.append(_format_arg(k, v))

    params["args"] = " ".join(args_parts)
    return params

# --- Build combinations ---
keys   = list(sweep_params.keys())
values = list(sweep_params.values())

if mode == "grid":
    combinations = list(itertools.product(*values))
elif mode == "zip":
    lengths = [len(v) for v in values]
    if len(set(lengths)) != 1:
        raise ValueError("zip mode: all sweep param lists must have the same length")
    combinations = list(zip(*values))
else:
    raise ValueError("mode must be 'grid' or 'zip'")

# Keep only meaningful cache/prefetch/cache-size combinations so experiment count
# stays focused and bounded.
if mode == "grid":
    key_to_idx = {k: i for i, k in enumerate(keys)}
    cm_idx = key_to_idx.get("--cache-mode")
    pf_idx = key_to_idx.get("--enable-prefetch")
    vt_idx = key_to_idx.get("--vram-thresh-gb")

    if cm_idx is not None and pf_idx is not None:
        control_vram = values[vt_idx][0] if vt_idx is not None else None
        filtered = []
        for combo in combinations:
            cache_mode = combo[cm_idx]
            prefetch = combo[pf_idx]
            vram_thresh = combo[vt_idx] if vt_idx is not None else None

            # With no cache, prefetch is a no-op and cache size is irrelevant.
            if cache_mode == "none":
                if prefetch:
                    continue
                if vt_idx is not None and vram_thresh != control_vram:
                    continue

            # warm_all pre-populates cache; async prefetch is redundant there.
            if cache_mode == "warm_all" and prefetch:
                continue

            filtered.append(combo)
        combinations = filtered

total_jobs = len(combinations)
if total_jobs > MAX_JOBS:
    raise ValueError(
        f"Filtered sweep still has {total_jobs} jobs (> {MAX_JOBS}). "
        "Narrow sweep_params or tighten non-redundant filtering."
    )
print(f"\ntotal jobs to generate: {total_jobs}")
print("\nJobs:")

for combo in combinations:
    p = build_job_params(fixed_params, keys, combo)
    p["profile_output"] = make_profile_output(profile_prefix, keys, combo)
    print(f"  {p['profile_output']}")
    print(f"    args: {p['args']}")

folders = [Path(slurm_dir), Path(config_dir)]
for folder in folders:
    if any(folder.iterdir()):
        raise RuntimeError("Output folder already has outputs - re-run setup")

confirm = input("do you want to continue? (y/n): ")
do_run = confirm.lower() == "y"

print("generating and submitting jobs...\n")

template = Path(TEMPLATE_PATH).read_text()
if "gpu_type" not in sweep_params and "gpu_type" not in fixed_params:
    template = template.replace(r'#SBATCH --constraint="{gpu_type}"', "")

for idx, combo in enumerate(combinations):
    job_params = build_job_params(fixed_params, keys, combo)
    job_params["profile_output"] = make_profile_output(profile_prefix, keys, combo)

    filled = template.format(**job_params)
    job_file = slurm_dir / f"job_{idx}.slurm"
    job_file.write_text(filled)

    cfg_file = config_dir / f"job_{idx}.cfg"
    with open(cfg_file, "w") as f:
        for k, v in job_params.items():
            f.write(f"{k}={v}\n")

    print(f"  job_{idx}: {job_params['profile_output']}")
    print(f"    args: {job_params['args']}")

    if do_run:
        result = subprocess.run(["sbatch", str(job_file)])
        print("  Submitted:", str(job_file))

print("\nAll jobs generated successfully.")
