#!/usr/bin/env python3
"""
nsys_kernel_table.py
--------------------
Given a directory of .nsys-rep files and a list of kernel name substrings,
produces a table where:
  - Each row    = one .nsys-rep file (identified by stem name)
  - Each column = one (kernel, metric) pair

All configuration is set in the PARAMETERS section below.
"""

import csv
import io
import subprocess
import sys
from pathlib import Path

import pandas as pd

# =============================================================================
# PARAMETERS — edit these
# =============================================================================

# Directory containing .nsys-rep files
NSYS_DIR = Path("/bigtemp/rhm4nj/gpu_arch/project/gsplat-fork/scripts/slurm/outputs/2026-03-16_17-28-40/results")

# Kernel substrings to search for (case-insensitive substring match on the Name column).
# The first match per file is used. Use short labels as keys for column headers.
KERNELS = {
    "rasterize_fwd":  "rasterizeToPixelsCUDA",
    "rasterize_bwd":  "rasterizeToPixelsBackwardCUDA",
    "preprocess":     "preprocessGaussiansCUDA",
    "adam":           "adam_update",
}

# Metrics to extract per kernel (must match column names in nsys cuda_gpu_kern_sum CSV output).
# Available columns: Time (%), Total Time (ns), Instances, Avg (ns), Med (ns), Min (ns), Max (ns), StdDev (ns)
METRICS = [
    "Total Time (ns)",
    "Instances",
]

# nsys binary (override if not on PATH, e.g. "/opt/nvidia/nsight-systems/2024.1/bin/nsys")
NSYS_BIN = "nsys"

# Output CSV path (None = only print to stdout)
OUTPUT_CSV = Path("nsys_kernel_table.csv")

# =============================================================================

REPORT = "cuda_gpu_kern_sum"


def run_nsys_stats(nsys_rep: Path) -> list[dict]:
    """Run nsys stats --format csv on a single .nsys-rep file and return parsed rows."""
    result = subprocess.run(
        [NSYS_BIN, "stats", "--format", "csv", "--report", REPORT,
         "--force-overwrite", "true", str(nsys_rep)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  [WARN] nsys failed for {nsys_rep.name}:\n{result.stderr.strip()}", file=sys.stderr)
        return []

    # nsys may emit multiple sections separated by blank lines / section headers.
    # The kernel summary CSV block starts after a header line containing "Time (%)".
    rows = []
    in_csv = False
    csv_lines = []

    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not in_csv:
            if stripped.startswith('"Time (%)"') or stripped.startswith("Time (%)"):
                in_csv = True
                csv_lines.append(stripped)
        else:
            if stripped == "" or stripped.startswith("="):
                break
            csv_lines.append(stripped)

    if not csv_lines:
        print(f"  [WARN] No kernel summary found in {nsys_rep.name}", file=sys.stderr)
        return []

    reader = csv.DictReader(io.StringIO("\n".join(csv_lines)))
    for row in reader:
        # Strip surrounding quotes that nsys sometimes leaves in field names
        rows.append({k.strip('"'): v.strip('"') for k, v in row.items()})

    return rows


def find_kernel(rows: list[dict], substring: str) -> dict | None:
    """Return the first row whose Name contains substring (case-insensitive)."""
    substring_lower = substring.lower()
    for row in rows:
        if substring_lower in row.get("Name", "").lower():
            return row
    return None


def build_table(nsys_dir: Path) -> pd.DataFrame:
    rep_files = sorted(nsys_dir.glob("*.nsys-rep"))
    if not rep_files:
        print(f"No .nsys-rep files found in {nsys_dir}", file=sys.stderr)
        sys.exit(1)

    # Multi-level columns: (kernel_label, metric)
    col_tuples = [(label, metric) for label in KERNELS for metric in METRICS]
    columns = pd.MultiIndex.from_tuples(col_tuples, names=["kernel", "metric"])

    records = {}
    for rep in rep_files:
        print(f"Processing {rep.name} ...")
        rows = run_nsys_stats(rep)
        row_data = {}
        for label, substring in KERNELS.items():
            matched = find_kernel(rows, substring)
            for metric in METRICS:
                key = (label, metric)
                if matched is None:
                    row_data[key] = None
                else:
                    raw = matched.get(metric, None)
                    # Convert to float if possible
                    try:
                        row_data[key] = float(raw.replace(",", "")) if raw else None
                    except (ValueError, AttributeError):
                        row_data[key] = raw
        records[rep.stem] = row_data

    df = pd.DataFrame.from_dict(records, orient="index", columns=columns)
    df.index.name = "profile"
    return df


def main():
    df = build_table(NSYS_DIR)

    # Print as a clean table
    print("\n" + "=" * 80)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:,.0f}".format)
    print(df.to_string())
    print("=" * 80 + "\n")

    if OUTPUT_CSV is not None:
        # Flatten multi-index columns to "kernel | metric" for CSV
        df_flat = df.copy()
        df_flat.columns = [f"{k} | {m}" for k, m in df_flat.columns]
        df_flat.to_csv(OUTPUT_CSV)
        print(f"Wrote table to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
