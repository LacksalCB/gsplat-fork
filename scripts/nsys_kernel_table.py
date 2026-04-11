#!/usr/bin/env python3
"""
nsys_kernel_table.py

Generate a summary table from .nsys-rep files for specified kernel functions.

Usage:
    python nsys_kernel_table.py <results_dir> <kernel1> [kernel2 ...] [options]

Each kernel argument is a substring matched (case-insensitive) against the
kernel Name field in nsys's cuda_gpu_kern_sum report.  If multiple rows match
a single pattern, they are aggregated: Time(%) and Total Time are summed;
Instances is summed; Avg/Med/Min/Max/StdDev come from the row with the largest
Total Time.

Output: a CSV (or printed table) where each row is one .nsys-rep file and the
columns are  <kernel_label>__<metric>  for every matched kernel × metric pair.

Options:
    --output FILE       Write CSV to FILE (default: stdout)
    --timeunit UNIT     ns | us | ms | s  (default: ms)
    --nsys PATH         Path to nsys binary (default: nsys in PATH)
    --force-export      Re-export .sqlite even if it already exists
"""

import argparse
import csv
import io
import subprocess
import sys
from pathlib import Path

# Metric display names (order preserved in output columns)
METRICS = ["Time (%)", "Total Time", "Instances", "Avg", "Med", "Min", "Max", "StdDev"]


# ---------------------------------------------------------------------------
# nsys helpers
# ---------------------------------------------------------------------------

def run_kern_sum(nsys: str, rep_file: Path, timeunit: str, force_export: bool) -> str:
    """
    Run  nsys stats -r cuda_gpu_kern_sum -f csv  on *rep_file* and return the
    raw stdout string.  nsys stats handles SQLite creation automatically.
    """
    cmd = [
        nsys, "stats",
        "-r", "cuda_gpu_kern_sum",
        "-f", "csv",
        f"--timeunit={timeunit}",
        "-o", "-",            # output to stdout
    ]
    if force_export:
        cmd.append("--force-export=true")
    cmd.append(str(rep_file))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout


def parse_kern_sum(raw: str) -> list[dict]:
    """
    Parse nsys cuda_gpu_kern_sum CSV output (stdout may contain progress lines
    before the CSV header).  Returns a list of row dicts.
    """
    lines = raw.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        # CSV header always starts with the Time column
        if line.startswith("Time (%)") or line.startswith('"Time (%)'):
            header_idx = i
            break
    if header_idx is None:
        return []
    csv_text = "\n".join(lines[header_idx:])
    reader = csv.DictReader(io.StringIO(csv_text))
    return list(reader)


# ---------------------------------------------------------------------------
# Matching & aggregation
# ---------------------------------------------------------------------------

def match_rows(rows: list[dict], pattern: str) -> list[dict]:
    """Return rows whose Name contains *pattern* (case-insensitive)."""
    pl = pattern.lower()
    return [r for r in rows if pl in r.get("Name", "").lower()]


def _col(row: dict, prefix: str) -> str | None:
    """Return the first column key that starts with *prefix*."""
    return next((k for k in row if k.startswith(prefix)), None)


def aggregate(matched: list[dict]) -> dict:
    """
    Collapse a list of matching rows into a single metrics dict.

    Summed  : Time (%), Total Time, Instances
    From dominant row (highest Total Time): Avg, Med, Min, Max, StdDev
    """
    if not matched:
        return {m: None for m in METRICS}

    time_col   = _col(matched[0], "Total Time")
    avg_col    = _col(matched[0], "Avg")
    med_col    = _col(matched[0], "Med")
    min_col    = _col(matched[0], "Min")
    max_col    = _col(matched[0], "Max")
    std_col    = _col(matched[0], "StdDev")

    def f(row, col):
        return float(row[col]) if col and row.get(col) not in (None, "") else None

    total_time = sum(f(r, time_col) or 0 for r in matched)
    instances  = sum(int(r["Instances"]) for r in matched)
    time_pct   = sum(float(r["Time (%)"]) for r in matched)

    dominant = max(matched, key=lambda r: f(r, time_col) or 0)
    return {
        "Time (%)":   round(time_pct, 4),
        "Total Time": total_time,
        "Instances":  instances,
        "Avg":        f(dominant, avg_col),
        "Med":        f(dominant, med_col),
        "Min":        f(dominant, min_col),
        "Max":        f(dominant, max_col),
        "StdDev":     f(dominant, std_col),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Build a per-.nsys-rep metric table for given kernel substrings."
    )
    ap.add_argument("results_dir", type=Path,
                    help="Directory containing .nsys-rep files")
    ap.add_argument("kernels", nargs="+",
                    help="Kernel name substrings (case-insensitive)")
    ap.add_argument("--output", "-o", type=Path, default=None,
                    help="Write CSV to this file (default: stdout)")
    ap.add_argument("--timeunit", default="ms",
                    choices=["ns", "us", "ms", "s"],
                    help="Time unit for reported values (default: ms)")
    ap.add_argument("--nsys", default="nsys",
                    help="Path to nsys binary")
    ap.add_argument("--force-export", action="store_true",
                    help="Force re-export of .sqlite from .nsys-rep")
    args = ap.parse_args()

    results_dir: Path = args.results_dir
    if not results_dir.is_dir():
        sys.exit(f"ERROR: {results_dir} is not a directory")

    rep_files = sorted(results_dir.glob("*.nsys-rep"))
    if not rep_files:
        sys.exit(f"ERROR: no .nsys-rep files found in {results_dir}")

    print(f"Found {len(rep_files)} .nsys-rep files", file=sys.stderr)

    # Sanitise pattern labels for column names
    def label(p: str) -> str:
        s = p.replace(" ", "_")
        return s[:50] if len(s) > 50 else s

    labels = [label(p) for p in args.kernels]

    # Column order: file | kernel1__metric1 | kernel1__metric2 | ... | kernelN__metricM
    col_headers = [f"{lbl}__{m}" for lbl in labels for m in METRICS]
    all_cols    = ["file"] + col_headers

    rows_out = []

    for rep_file in rep_files:
        print(f"  {rep_file.name} ...", file=sys.stderr, end=" ", flush=True)
        try:
            raw       = run_kern_sum(args.nsys, rep_file, args.timeunit, args.force_export)
            kern_rows = parse_kern_sum(raw)
        except RuntimeError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            continue

        row = {"file": rep_file.stem}
        for pattern, lbl in zip(args.kernels, labels):
            matched = match_rows(kern_rows, pattern)
            n = len(matched)
            agg = aggregate(matched)
            for m in METRICS:
                row[f"{lbl}__{m}"] = agg[m]
            if n == 0:
                print(f"[no match: '{pattern}']", file=sys.stderr, end=" ")

        rows_out.append(row)
        print("done", file=sys.stderr)

    if not rows_out:
        sys.exit("ERROR: no data collected")

    def write_csv(f):
        writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows_out)

    if args.output:
        with open(args.output, "w", newline="") as f:
            write_csv(f)
        print(f"\nWrote {len(rows_out)} rows → {args.output}", file=sys.stderr)
    else:
        write_csv(sys.stdout)


if __name__ == "__main__":
    main()
