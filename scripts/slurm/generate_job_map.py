#!/usr/bin/env python3
"""
Generate job_map.json for existing output folders that don't have one.

Usage:
    # Single folder
    python generate_job_map.py outputs/2026-03-16_18-51-37

    # All folders under outputs/
    python generate_job_map.py --all

    # Custom outputs root
    python generate_job_map.py --all --root /path/to/outputs
"""

import json
import re
import sys
from pathlib import Path

OUTPUTS_ROOT = Path(__file__).parent / "outputs"


def parse_cfg(cfg_path: Path) -> dict:
    """Parse a job_N.cfg into a dict. Values may span multiple lines (e.g. module_loads)."""
    result = {}
    current_key = None
    for line in cfg_path.read_text().splitlines():
        if "=" in line and not line.startswith(" ") and not line.startswith("\t"):
            key, _, val = line.partition("=")
            result[key.strip()] = val.strip()
            current_key = key.strip()
        elif current_key:
            # continuation line (e.g. second line of module_loads)
            result[current_key] += "\n" + line
    return result


def profile_output_from_slurm(slurm_path: Path) -> str | None:
    """Extract the nsys --output value from a .slurm script.
    Looks only within the nsys profile call to avoid matching #SBATCH --output."""
    text = slurm_path.read_text()
    # Isolate everything after the first 'nsys profile' invocation
    nsys_idx = text.find("nsys profile")
    if nsys_idx == -1:
        return None
    nsys_section = text[nsys_idx:]
    match = re.search(r"--output[= ]([^\s\\]+)", nsys_section)
    if not match:
        return None
    val = match.group(1)
    # Skip values that contain unresolved shell variables
    if "$" in val or "%" in val:
        return None
    return Path(val).name  # basename only


def make_job_map(run_dir: Path) -> dict | None:
    config_dir = run_dir / "configs"
    slurm_dir = run_dir / "slurm_scripts"

    if not config_dir.exists():
        return None

    cfg_files = sorted(config_dir.glob("job_*.cfg"),
                       key=lambda p: int(re.search(r"\d+", p.stem).group()))
    if not cfg_files:
        return None

    job_map = {}
    for cfg_path in cfg_files:
        job_id = cfg_path.stem  # e.g. "job_0"
        params = parse_cfg(cfg_path)
        profile_output = params.get("profile_output", "")

        # Old runs stored "${SLURM_JOB_ID}" — unusable, fall back to slurm script
        if not profile_output or "${SLURM_JOB_ID}" in profile_output:
            slurm_path = slurm_dir / f"{job_id}.slurm"
            profile_output = profile_output_from_slurm(slurm_path) if slurm_path.exists() else None

        if profile_output:
            # Strip to basename if it's a full path
            profile_output = Path(profile_output).name

        job_map[job_id] = profile_output or "UNKNOWN"

    return job_map


def process_run(run_dir: Path, overwrite: bool = False) -> bool:
    out_path = run_dir / "job_map.json"
    if out_path.exists() and not overwrite:
        print(f"  [skip] {run_dir.name} — job_map.json already exists")
        return False

    job_map = make_job_map(run_dir)
    if job_map is None:
        print(f"  [skip] {run_dir.name} — no configs/ directory or no job_*.cfg files")
        return False

    out_path.write_text(json.dumps(job_map, indent=2))
    print(f"  [ok]   {run_dir.name} — wrote {len(job_map)} entries → {out_path}")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate job_map.json for SLURM output folders")
    parser.add_argument("run_dir", nargs="?", help="Path to a single run folder")
    parser.add_argument("--all", action="store_true", help="Process all folders under --root")
    parser.add_argument("--root", type=Path, default=OUTPUTS_ROOT, help="Root outputs directory (default: ./outputs)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing job_map.json files")
    args = parser.parse_args()

    if args.all:
        root = args.root
        if not root.exists():
            print(f"Error: outputs root not found: {root}", file=sys.stderr)
            sys.exit(1)
        run_dirs = sorted(d for d in root.iterdir() if d.is_dir())
        print(f"Scanning {len(run_dirs)} folders under {root}\n")
        for run_dir in run_dirs:
            process_run(run_dir, overwrite=args.overwrite)
    elif args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            print(f"Error: folder not found: {run_dir}", file=sys.stderr)
            sys.exit(1)
        process_run(run_dir, overwrite=args.overwrite)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
