#!/usr/bin/env python3
"""
nsys_kern_table.py
------------------
Given a directory of .nsys-rep files and a list of kernel name patterns,
generates a table where:
  - rows    = each .nsys-rep (by stem name)
  - columns = (kernel_pattern, metric) multi-index

When a pattern matches multiple kernel rows (e.g. different template
instantiations), metrics are aggregated:
  - Total Time (ns), Instances, Time (%) : summed
  - Avg (ns)                             : total_time / total_instances
  - Min (ns)                             : minimum across matches
  - Max (ns)                             : maximum across matches
  - Med (ns), StdDev (ns)               : taken from the largest contributor

Requires: nsys on PATH, pandas

USAGE:
  python3 nsys_kern_table.py \\
      --dir /path/to/results \\
      --kernels rasterize_to_pixels_3dgs_bwd rasterize_to_pixels_3dgs_fwd \\
      [--metrics "Total Time (ns)" Instances "Avg (ns)"] \\
      [--output kern_table.csv] \\
      [--force-export] \\
      [--workers 4]

Available metrics (default: Total Time (ns), Instances, Avg (ns)):
  Time (%), Total Time (ns), Instances, Avg (ns), Med (ns), Min (ns), Max (ns), StdDev (ns)
"""

import argparse
import io
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


ALL_METRICS = ["Time (%)", "Total Time (ns)", "Instances", "Avg (ns)", "Med (ns)", "Min (ns)", "Max (ns)", "StdDev (ns)"]
SUMMED_METRICS = {"Time (%)", "Total Time (ns)", "Instances"}
DEFAULT_METRICS = ["Total Time (ns)", "Instances", "Avg (ns)"]


def run_nsys_stats(nsys_rep: Path, force_export: bool) -> pd.DataFrame | None:
    """Run `nsys stats -r cuda_gpu_kern_sum -f csv` and return the result as a DataFrame."""
    cmd = ["nsys", "stats", "-r", "cuda_gpu_kern_sum", "-f", "csv"]
    if force_export:
        cmd.append("--force-export=true")
    cmd.append(str(nsys_rep))

    result = subprocess.run(cmd, capture_output=True, text=True)

    # nsys writes progress/notices to stderr, CSV to stdout;
    # search both in case of version differences
    combined = result.stdout + "\n" + result.stderr
    lines = combined.splitlines()

    csv_start = next((i for i, ln in enumerate(lines) if ln.startswith("Time (%),Total Time")), None)
    if csv_start is None:
        return None

    csv_text = "\n".join(lines[csv_start:])
    try:
        return pd.read_csv(io.StringIO(csv_text))
    except Exception as e:
        print(f"  [WARN] CSV parse error for {nsys_rep.name}: {e}", file=sys.stderr)
        return None


def aggregate(matched: pd.DataFrame, metrics: list[str]) -> dict:
    """Aggregate rows that matched a single kernel pattern."""
    result = {}
    for m in metrics:
        if m not in matched.columns:
            result[m] = None
            continue
        if m in SUMMED_METRICS:
            result[m] = float(matched[m].sum())
        elif m == "Avg (ns)":
            total_inst = matched["Instances"].sum()
            result[m] = float(matched["Total Time (ns)"].sum() / total_inst) if total_inst else None
        elif m == "Min (ns)":
            result[m] = float(matched[m].min())
        elif m == "Max (ns)":
            result[m] = float(matched[m].max())
        else:
            # Med (ns), StdDev (ns): take from the row with the largest total time
            biggest = matched["Total Time (ns)"].idxmax()
            result[m] = float(matched.loc[biggest, m])
    return result


def process_file(nsys_rep: Path, kernels: list[str], metrics: list[str], force_export: bool) -> tuple[str, dict]:
    print(f"Processing {nsys_rep.name} ...", file=sys.stderr)
    df = run_nsys_stats(nsys_rep, force_export)

    row_data: dict = {}
    for pattern in kernels:
        if df is not None:
            mask = df["Name"].str.contains(pattern, regex=False, na=False)
            matched = df[mask]
            kern_metrics = aggregate(matched, metrics) if not matched.empty else {m: None for m in metrics}
        else:
            kern_metrics = {m: None for m in metrics}

        for m, v in kern_metrics.items():
            row_data[(pattern, m)] = v

    return nsys_rep.stem, row_data


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dir", required=True, type=Path,
                        help="Directory containing .nsys-rep files")
    parser.add_argument("--kernels", required=True, nargs="+",
                        help="Kernel name substring patterns to match (case-sensitive)")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS,
                        metavar="METRIC",
                        help=f"Metrics to report (default: {DEFAULT_METRICS}). "
                             f"Available: {ALL_METRICS}")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write flat CSV to this path in addition to console output")
    parser.add_argument("--force-export", action="store_true",
                        help="Force re-export of SQLite even if it already exists")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel nsys processes (default: 4)")
    args = parser.parse_args()

    nsys_reps = sorted(args.dir.glob("*.nsys-rep"))
    if not nsys_reps:
        print(f"No .nsys-rep files found in {args.dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(nsys_reps)} .nsys-rep files", file=sys.stderr)

    col_tuples = [(k, m) for k in args.kernels for m in args.metrics]
    columns = pd.MultiIndex.from_tuples(col_tuples, names=["kernel", "metric"])

    records: dict = {}
    with ThreadPoolExecutor(max_workers=min(args.workers, len(nsys_reps))) as executor:
        futures = {
            executor.submit(process_file, f, args.kernels, args.metrics, args.force_export): f
            for f in nsys_reps
        }
        for future in as_completed(futures):
            stem, row_data = future.result()
            records[stem] = row_data

    df_out = pd.DataFrame.from_dict(records, orient="index", columns=columns)
    df_out.index.name = "profile"
    df_out.sort_index(inplace=True)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 400)
    pd.set_option("display.float_format", "{:,.1f}".format)
    print("\n" + "=" * 120)
    print(df_out.to_string())
    print("=" * 120 + "\n")

    if args.output:
        df_flat = df_out.copy()
        df_flat.columns = [f"{k} | {m}" for k, m in df_flat.columns]
        df_flat.to_csv(args.output)
        print(f"Wrote table to: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
