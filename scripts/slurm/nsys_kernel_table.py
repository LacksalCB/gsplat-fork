#!/usr/bin/env python3
"""
nsys_kernel_table.py
--------------------
Given a directory of .nsys-rep files and a list of kernel name substrings,
produces a table where:
  - Each row    = one .nsys-rep file (identified by stem name)
  - Each column = one (kernel, metric) pair

All configuration is set in the PARAMETERS section below.

USAGE:
  Step 1 — Export all .nsys-rep files to .sqlite (do this once):
      for f in *.nsys-rep; do
          nsys export --type sqlite --force-overwrite true "$f" &
      done
      wait

  Step 2 — Run this script (fast, no nsys subprocess per file):
      python3 nsys_kernel_table.py
"""

import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

# =============================================================================
# PARAMETERS — edit these
# =============================================================================

# Directory containing .sqlite files (exported from .nsys-rep)
NSYS_DIR = Path("/scratch/rhm4nj/gpu_arch/gsplat-fork/scripts/slurm/outputs/2026-03-19_05-49-44/results")

# Kernel substrings to search for (case-insensitive substring match on kernel name).
# The first match per file is used. Keys are used as column headers.
KERNELS = {
    "multi_tensor_apply_kernel": "multi_tensor_apply_kernel",
    "raster_bwd":                "rasterize_to_pixels_3dgs_bwd_kernel",
    "cat_copy":                  "CatArrayBatchedCopy",
    "elementwise_kernel":        "elementwise_kernel",
    "raster_fwd":                "rasterize_to_pixels_3dgs_fwd_kernel",
}

# Metrics to extract per kernel.
# Available: "Total Time (ns)", "Instances", "Avg (ns)", "Min (ns)", "Max (ns)"
METRICS = [
    "Total Time (ns)",
    "Instances",
]

# Output CSV path (None = only print to stdout)
OUTPUT_CSV = Path("nsys_kernel_table.csv")

# Max parallel workers for querying sqlite files
MAX_WORKERS = 8

# =============================================================================

SQLITE_QUERY = """
    SELECT
        s.value                    AS Name,
        SUM(k.end - k.start)       AS "Total Time (ns)",
        COUNT(*)                   AS "Instances",
        AVG(k.end - k.start)       AS "Avg (ns)",
        MIN(k.end - k.start)       AS "Min (ns)",
        MAX(k.end - k.start)       AS "Max (ns)"
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds s ON s.id = k.shortName
    GROUP BY k.shortName
    ORDER BY "Total Time (ns)" DESC
"""


def query_sqlite(sqlite_path: Path) -> list[dict]:
    """Query kernel summary directly from a .sqlite file exported by nsys."""
    try:
        con = sqlite3.connect(sqlite_path)
        cur = con.execute(SQLITE_QUERY)
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, row)) for row in cur.fetchall()]
        con.close()
        return rows
    except Exception as e:
        print(f"  [WARN] Failed to query {sqlite_path.name}: {e}", file=sys.stderr)
        return []


def find_kernel(rows: list[dict], substring: str) -> dict | None:
    """Return the first row whose Name contains substring (case-insensitive)."""
    substring_lower = substring.lower()
    for row in rows:
        if substring_lower in row.get("Name", "").lower():
            return row
    return None


def process(sqlite_path: Path) -> tuple[str, dict]:
    """Query one sqlite file and return (stem, row_data)."""
    print(f"Processing {sqlite_path.name} ...")
    rows = query_sqlite(sqlite_path)
    if not rows:
        print(f"  [WARN] No kernel data found in {sqlite_path.name}", file=sys.stderr)

    row_data = {}
    for label, substring in KERNELS.items():
        matched = find_kernel(rows, substring)
        for metric in METRICS:
            key = (label, metric)
            if matched is None:
                row_data[key] = None
            else:
                val = matched.get(metric, None)
                try:
                    row_data[key] = float(val) if val is not None else None
                except (ValueError, TypeError):
                    row_data[key] = val
    return sqlite_path.stem, row_data


def build_table(nsys_dir: Path) -> pd.DataFrame:
    sqlite_files = sorted(nsys_dir.glob("*.sqlite"))

    if not sqlite_files:
        print(f"No .sqlite files found in {nsys_dir}", file=sys.stderr)
        print("Run this first to export your .nsys-rep files:", file=sys.stderr)
        print("  for f in *.nsys-rep; do nsys export --type sqlite --force-overwrite true \"$f\" & done && wait", file=sys.stderr)
        sys.exit(1)

    col_tuples = [(label, metric) for label in KERNELS for metric in METRICS]
    columns = pd.MultiIndex.from_tuples(col_tuples, names=["kernel", "metric"])

    records = {}
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(sqlite_files))) as executor:
        futures = {executor.submit(process, f): f for f in sqlite_files}
        for future in as_completed(futures):
            stem, row_data = future.result()
            records[stem] = row_data

    df = pd.DataFrame.from_dict(records, orient="index", columns=columns)
    df.index.name = "profile"
    df.sort_index(inplace=True)
    return df


def main():
    df = build_table(NSYS_DIR)

    print("\n" + "=" * 80)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:,.0f}".format)
    print(df.to_string())
    print("=" * 80 + "\n")

    if OUTPUT_CSV is not None:
        df_flat = df.copy()
        df_flat.columns = [f"{k} | {m}" for k, m in df_flat.columns]
        df_flat.to_csv(OUTPUT_CSV)
        print(f"Wrote table to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()