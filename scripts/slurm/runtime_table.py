#!/usr/bin/env python3
"""
Parse real runtimes from SLURM .err files and print a table.

Usage:
    python3 runtime_table.py <run_dir>               # use profile names (default)
    python3 runtime_table.py <run_dir> --job-ids     # use job_N labels instead
    python3 runtime_table.py <run_dir> --csv         # output CSV

Example:
    python3 runtime_table.py saved_outputs/2026-03-19_02-31-44_lru_frustrum_2
"""

import argparse
import re
import sys
from pathlib import Path


def parse_real_time(err_text: str) -> str | None:
    """Extract the 'real' runtime line, e.g. '19m54.702s'."""
    match = re.search(r"^real\s+(\S+)", err_text, re.MULTILINE)
    return match.group(1) if match else None


def real_to_seconds(time_str: str) -> float | None:
    """Convert '19m54.702s' or '1h2m3.4s' to total seconds for sorting."""
    if time_str is None:
        return None
    m = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(\d+(?:\.\d+)?)s", time_str)
    if not m:
        return None
    h = int(m.group(1) or 0)
    mins = int(m.group(2) or 0)
    secs = float(m.group(3))
    return h * 3600 + mins * 60 + secs


def parse_cfg(cfg_path: Path) -> dict:
    """Parse key=value cfg file (handles multi-line values like module_loads)."""
    result = {}
    current_key = None
    for line in cfg_path.read_text().splitlines():
        if "=" in line and not line.startswith((" ", "\t")):
            key, _, val = line.partition("=")
            result[key.strip()] = val.strip()
            current_key = key.strip()
        elif current_key:
            result[current_key] += "\n" + line
    return result


def get_profile_name(cfg_path: Path) -> str:
    """Return the basename of profile_output from a .cfg, or the job id if missing."""
    if not cfg_path.exists():
        return cfg_path.stem  # fallback: job_N
    params = parse_cfg(cfg_path)
    profile_output = params.get("profile_output", "")
    if not profile_output or "${SLURM_JOB_ID}" in profile_output:
        return cfg_path.stem
    return Path(profile_output).name


def collect_rows(run_dir: Path, use_job_ids: bool) -> list[dict]:
    err_dir = run_dir / "slurm_err"
    cfg_dir = run_dir / "configs"

    if not err_dir.exists():
        print(f"Error: no slurm_err/ directory in {run_dir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for err_file in sorted(err_dir.glob("job_*.err"),
                           key=lambda p: int(re.search(r"job_(\d+)", p.name).group(1))):

        job_num = int(re.search(r"job_(\d+)", err_file.name).group(1))
        job_key = f"job_{job_num}"

        cfg_path = cfg_dir / f"{job_key}.cfg"
        label = job_key if use_job_ids else get_profile_name(cfg_path)

        text = err_file.read_text()
        real_time = parse_real_time(text)
        total_secs = real_to_seconds(real_time)

        rows.append({
            "job": job_key,
            "label": label,
            "real": real_time or "N/A",
            "seconds": total_secs,
        })

    return rows


def print_table(rows: list[dict], use_job_ids: bool) -> None:
    label_col = "job" if use_job_ids else "profile"
    col1_width = max(len(label_col), max(len(r["label"]) for r in rows))
    real_width = max(len("real"), max(len(r["real"]) for r in rows))

    header = f"{'#':<4}  {label_col:<{col1_width}}  {'real':>{real_width}}"
    sep    = f"{'─'*4}  {'─'*col1_width}  {'─'*real_width}"
    print(header)
    print(sep)
    for i, r in enumerate(rows):
        print(f"{i:<4}  {r['label']:<{col1_width}}  {r['real']:>{real_width}}")


def print_csv(rows: list[dict], use_job_ids: bool) -> None:
    label_col = "job" if use_job_ids else "profile"
    print(f"job,{label_col},real,seconds")
    for r in rows:
        print(f"{r['job']},{r['label']},{r['real']},{r['seconds'] if r['seconds'] is not None else ''}")


def main():
    parser = argparse.ArgumentParser(description="Show real runtimes from SLURM .err files")
    parser.add_argument("run_dir", type=Path, help="Path to the run folder (contains slurm_err/ and configs/)")
    parser.add_argument("--job-ids", action="store_true", help="Use job_N labels instead of profile names")
    parser.add_argument("--csv", action="store_true", help="Output as CSV")
    args = parser.parse_args()

    rows = collect_rows(args.run_dir, use_job_ids=args.job_ids)
    if not rows:
        print("No .err files found.", file=sys.stderr)
        sys.exit(1)

    if args.csv:
        print_csv(rows, use_job_ids=args.job_ids)
    else:
        print_table(rows, use_job_ids=args.job_ids)


if __name__ == "__main__":
    main()
