#!/usr/bin/env python3
"""
nsys_mem_table.py
-----------------
Queries memory operation stats (Memset, HtoD, DtoH, DtoD) directly from
.sqlite files exported by nsys.

  - CUPTI_ACTIVITY_KIND_MEMCPY  (copyKind: 1=HtoD, 2=DtoH, 8=DtoD)
  - CUPTI_ACTIVITY_KIND_MEMSET  (has direct 'bytes' column)

Reports Total Time (ns), Instances, and Bandwidth (GB/s) per operation.

USAGE:
  Step 1 — Export all .nsys-rep files to .sqlite (do this once):
      for f in *.nsys-rep; do
          nsys export --type sqlite --force-overwrite true "$f" &
      done
      wait

  Step 2 — Run this script:
      python3 nsys_mem_table.py
"""

import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

# =============================================================================
# PARAMETERS — edit these
# =============================================================================

NSYS_DIR   = Path("/scratch/rhm4nj/gpu_arch/gsplat-fork/scripts/slurm/outputs/2026-03-19_05-49-44_ablations/results")
OUTPUT_CSV = Path("nsys_mem_table.csv")
MAX_WORKERS = 8

# =============================================================================

COPY_KINDS = {1: "HtoD", 2: "DtoH", 8: "DtoD"}

MEMCPY_QUERY = """
    SELECT
        copyKind,
        SUM(end - start)  AS total_time_ns,
        COUNT(*)          AS instances,
        SUM(bytes)        AS total_bytes
    FROM CUPTI_ACTIVITY_KIND_MEMCPY
    GROUP BY copyKind
"""

MEMSET_QUERY = """
    SELECT
        SUM(end - start)  AS total_time_ns,
        COUNT(*)          AS instances,
        SUM(bytes)        AS total_bytes
    FROM CUPTI_ACTIVITY_KIND_MEMSET
"""

METRICS = ["Total Time (ns)", "Instances", "Bandwidth (GB/s)"]
OPS     = ["Memset", "HtoD", "DtoH", "DtoD"]


def compute_bandwidth(total_bytes, total_time_ns):
    """Returns GB/s given total bytes and total duration in nanoseconds."""
    if not total_bytes or not total_time_ns or total_time_ns == 0:
        return None
    return (total_bytes / 1e9) / (total_time_ns * 1e-9)


def query_sqlite(sqlite_path: Path) -> dict:
    result = {op: {m: None for m in METRICS} for op in OPS}
    try:
        con = sqlite3.connect(sqlite_path)

        # --- memcpy ---
        try:
            for row in con.execute(MEMCPY_QUERY).fetchall():
                kind_name = COPY_KINDS.get(int(row[0]))
                if kind_name:
                    total_time_ns, instances, total_bytes = row[1], row[2], row[3]
                    result[kind_name]["Total Time (ns)"]  = float(total_time_ns) if total_time_ns is not None else None
                    result[kind_name]["Instances"]        = float(instances)     if instances     is not None else None
                    result[kind_name]["Bandwidth (GB/s)"] = compute_bandwidth(total_bytes, total_time_ns)
        except Exception as e:
            print(f"  [WARN] memcpy query failed in {sqlite_path.name}: {e}", file=sys.stderr)

        # --- memset ---
        try:
            row = con.execute(MEMSET_QUERY).fetchone()
            if row:
                total_time_ns, instances, total_bytes = row[0], row[1], row[2]
                result["Memset"]["Total Time (ns)"]  = float(total_time_ns) if total_time_ns is not None else None
                result["Memset"]["Instances"]        = float(instances)     if instances     is not None else None
                result["Memset"]["Bandwidth (GB/s)"] = compute_bandwidth(total_bytes, total_time_ns)
        except Exception as e:
            print(f"  [WARN] memset query failed in {sqlite_path.name}: {e}", file=sys.stderr)

        con.close()
    except Exception as e:
        print(f"  [WARN] Failed to open {sqlite_path.name}: {e}", file=sys.stderr)

    return result


def process(sqlite_path: Path) -> tuple[str, dict]:
    print(f"Processing {sqlite_path.name} ...")
    op_data = query_sqlite(sqlite_path)
    row_data = {}
    for op in OPS:
        for metric in METRICS:
            row_data[(op, metric)] = op_data[op][metric]
    return sqlite_path.stem, row_data


def build_table(nsys_dir: Path) -> pd.DataFrame:
    sqlite_files = sorted(nsys_dir.glob("*.sqlite"))
    if not sqlite_files:
        print(f"No .sqlite files found in {nsys_dir}", file=sys.stderr)
        print("Export first:", file=sys.stderr)
        print("  for f in *.nsys-rep; do nsys export --type sqlite --force-overwrite true \"$f\" & done && wait", file=sys.stderr)
        sys.exit(1)

    col_tuples = [(op, m) for op in OPS for m in METRICS]
    columns    = pd.MultiIndex.from_tuples(col_tuples, names=["operation", "metric"])

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

    print("\n" + "=" * 120)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)
    pd.set_option("display.float_format", "{:,.2f}".format)
    print(df.to_string())
    print("=" * 120 + "\n")

    if OUTPUT_CSV is not None:
        df_flat = df.copy()
        df_flat.columns = [f"{op} | {m}" for op, m in df_flat.columns]
        df_flat.to_csv(OUTPUT_CSV)
        print(f"Wrote table to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()