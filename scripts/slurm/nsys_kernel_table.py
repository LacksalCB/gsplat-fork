#!/usr/bin/env python3

"""
nsys_kernel_table.py
--------------------
Queries memory operation stats (Memset, HtoD, DtoH, DtoD) and GPU utilization
directly from .sqlite files exported by nsys.

  - CUPTI_ACTIVITY_KIND_MEMCPY  (copyKind: 1=HtoD, 2=DtoH, 8=DtoD)
  - CUPTI_ACTIVITY_KIND_MEMSET  (has direct 'bytes' column)
  - CUPTI_ACTIVITY_KIND_KERNEL  (used for GPU utilization %)

Reports Total Time (ns), Instances, and Bandwidth (GB/s) per operation,
plus GPU Utilization (%).

USAGE:
  Step 1 — Export all .nsys-rep files to .sqlite (do this once):
      ./nsys_export_sqlite.sh <results_dir>

  Step 2 — Run this script:
      python3 nsys_kernel_table.py <results_dir> [--output output.csv]

  Examples:
      python3 nsys_kernel_table.py outputs/2026-03-25_23-49-37_ablations_p2/results
      python3 nsys_kernel_table.py outputs/2026-03-25_23-49-37_ablations_p2/results --output my_table.csv
"""

import argparse
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

MAX_WORKERS = 8
# MAX_WORKERS = 4
# MAX_WORKERS = 12

COPY_KINDS = {1: "HtoD", 2: "DtoH", 8: "DtoD"}

MEMCPY_QUERY = """
    SELECT
        copyKind,
        SUM(end - start) AS total_time_ns,
        COUNT(*) AS instances,
        SUM(bytes) AS total_bytes
    FROM CUPTI_ACTIVITY_KIND_MEMCPY
    GROUP BY copyKind
"""

MEMSET_QUERY = """
    SELECT
        SUM(end - start) AS total_time_ns,
        COUNT(*) AS instances,
        SUM(bytes) AS total_bytes
    FROM CUPTI_ACTIVITY_KIND_MEMSET
"""

GPU_UTIL_QUERY = """
    SELECT
        SUM(end - start) AS total_kernel_ns,
        MAX(end) - MIN(start) AS span_ns
    FROM CUPTI_ACTIVITY_KIND_KERNEL
"""

VRAM_QUERY = """
    SELECT bytes, isAlloc
    FROM CUPTI_ACTIVITY_KIND_MEMORY2
    ORDER BY timestamp
"""

METRICS = ["Total Time (ns)", "Instances", "Bandwidth (GB/s)"]
OPS = ["Memset", "HtoD", "DtoH", "DtoD"]
GPU_METRICS = ["Utilization (%)", "Peak VRAM (MB)"]
GPU_OPS = ["GPU"]


def compute_bandwidth(total_bytes, total_time_ns):
    """Returns GB/s given total bytes and total duration in nanoseconds."""
    if not total_bytes or not total_time_ns or total_time_ns == 0:
        return None
    return (total_bytes / 1e9) / (total_time_ns * 1e-9)


def query_sqlite(sqlite_path: Path) -> tuple[dict, dict]:
    mem_result = {op: {m: None for m in METRICS} for op in OPS}
    gpu_result = {op: {m: None for m in GPU_METRICS} for op in GPU_OPS}
    try:
        con = sqlite3.connect(sqlite_path)

        try:
            for row in con.execute(MEMCPY_QUERY).fetchall():
                kind_name = COPY_KINDS.get(int(row[0]))
                if kind_name:
                    total_time_ns, instances, total_bytes = row[1], row[2], row[3]
                    mem_result[kind_name]["Total Time (ns)"] = float(total_time_ns) if total_time_ns is not None else None
                    mem_result[kind_name]["Instances"] = float(instances) if instances is not None else None
                    mem_result[kind_name]["Bandwidth (GB/s)"] = compute_bandwidth(total_bytes, total_time_ns)
        except Exception as e:
            print(f"  warn: memcpy query failed in {sqlite_path.name}: {e}", file=sys.stderr) # short warning text

        try:
            row = con.execute(MEMSET_QUERY).fetchone()
            if row:
                total_time_ns, instances, total_bytes = row[0], row[1], row[2]
                mem_result["Memset"]["Total Time (ns)"] = float(total_time_ns) if total_time_ns is not None else None
                mem_result["Memset"]["Instances"] = float(instances) if instances is not None else None
                mem_result["Memset"]["Bandwidth (GB/s)"] = compute_bandwidth(total_bytes, total_time_ns)
        except Exception as e:
            print(f"  warn: memset query failed in {sqlite_path.name}: {e}", file=sys.stderr)

        try:
            row = con.execute(GPU_UTIL_QUERY).fetchone()
            if row and row[0] is not None and row[1] and row[1] > 0:
                # gpuUtil = (float(row[0]) / float(row[1])) * 100.0
                gpu_result["GPU"]["Utilization (%)"] = float(row[0]) / float(row[1]) * 100.0
        except Exception as e:
            print(f"  warn: gpu util query failed in {sqlite_path.name}: {e}", file=sys.stderr)

        try:
            rows = con.execute(VRAM_QUERY).fetchall()
            running, peak = 0, 0
            for bytes_, is_alloc in rows:
                if bytes_ is None:
                    continue
                running += bytes_ if is_alloc else -bytes_
                if running > peak:
                    peak = running
            if peak > 0:
                gpu_result["GPU"]["Peak VRAM (MB)"] = peak / 1e6
        except Exception as e:
            print(f"  warn: vram query failed in {sqlite_path.name}: {e}", file=sys.stderr)

        con.close()
    except Exception as e:
        print(f"  warn: failed to open {sqlite_path.name}: {e}", file=sys.stderr)

    return mem_result, gpu_result


def processSqlite(sqlite_path: Path) -> tuple[str, dict]:
    print(f"processing {sqlite_path.name} ...")
    mem_data, gpu_data = query_sqlite(sqlite_path)
    row_data = {}
    for op in OPS:
        for metric in METRICS:
            row_data[(op, metric)] = mem_data[op][metric]
    for op in GPU_OPS:
        for metric in GPU_METRICS:
            row_data[(op, metric)] = gpu_data[op][metric]
    return sqlite_path.stem, row_data


def build_table(nsys_dir: Path) -> pd.DataFrame:
    sqlite_files = sorted(nsys_dir.glob("*.sqlite"))
    if not sqlite_files:
        print(f"no .sqlite files found in {nsys_dir}", file=sys.stderr)
        print("export first:", file=sys.stderr)
        print("  ./nsys_export_sqlite.sh <results_dir>", file=sys.stderr)
        sys.exit(1)

    col_tuples = [(op, m) for op in OPS for m in METRICS] + [(op, m) for op in GPU_OPS for m in GPU_METRICS]
    columns = pd.MultiIndex.from_tuples(col_tuples, names=["operation", "metric"])

    records = {}
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(sqlite_files))) as executor:
        futures = {executor.submit(processSqlite, f): f for f in sqlite_files}
        for future in as_completed(futures):
            stem, row_data = future.result()
            records[stem] = row_data

    df = pd.DataFrame.from_dict(records, orient="index", columns=columns)
    df.index.name = "profile"
    df.sort_index(inplace=True)
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", type=Path,
                        help="Directory containing .sqlite files")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output CSV path (default: <results_dir>/nsys_mem_table.csv)")
    args = parser.parse_args()

    nsys_dir = args.results_dir
    output_csv = args.output if args.output is not None else nsys_dir / "nsys_mem_table.csv"

    df = build_table(nsys_dir)

    print("\n" + "=" * 120)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)
    pd.set_option("display.float_format", "{:,.2f}".format)
    print(df.to_string())
    print("=" * 120 + "\n")

    df_flat = df.copy()
    df_flat.columns = [f"{op} | {m}" for op, m in df_flat.columns]
    df_flat.to_csv(output_csv)
    print(f"wrote table to: {output_csv}") # keep logs readable and plain


if __name__ == "__main__":
    main()
