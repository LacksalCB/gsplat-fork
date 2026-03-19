#!/usr/bin/env python3
# Usage: python aggregate_stats.py /path/to/output-dir
# Parses slurm_out/*.out files and writes stats.csv to the output dir root.

import csv
import json
import re
import sys
from pathlib import Path


def parse_out_file(out_file: Path) -> list[dict]:
    """Extract all train/val stat rows from a single .out file."""
    rows = []
    lines = out_file.read_text(errors="replace").splitlines()

    scene = None
    current_section = None  # "val" or "train"
    pending_path = None

    for line in lines:
        line = line.strip()

        # Detect scene name — only match single lowercase words (e.g. "Running garden")
        # to avoid matching trainer log lines like "Running evaluation..."
        if scene is None:
            m = re.match(r"^Running ([a-z_]+)$", line)
            if m:
                scene = m.group(1)
                continue

        if line == "=== Eval Stats ===":
            current_section = "val"
            continue
        if line == "=== Train Stats ===":
            current_section = "train"
            continue

        # Detect a stats filepath line (e.g. results/benchmark/garden/stats/val_step29999.json)
        if current_section and line.endswith(".json") and "stats/" in line:
            pending_path = line
            continue

        # Detect the JSON line immediately following a filepath
        if pending_path and line.startswith("{"):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                pending_path = None
                continue

            # Extract step from filename
            fname = Path(pending_path).name  # e.g. val_step29999.json
            step_m = re.search(r"step(\d+)", fname)
            step = int(step_m.group(1)) if step_m else None

            row = {
                "scene": scene,
                "split": current_section,
                "step": step,
                "psnr": data.get("psnr"),
                "ssim": data.get("ssim"),
                "lpips": data.get("lpips"),
                "ellipse_time": data.get("ellipse_time"),
                "num_GS": data.get("num_GS"),
                "mem_gb": data.get("mem"),
            }
            rows.append(row)
            pending_path = None

    return rows


def main():
    if len(sys.argv) < 2:
        print("Usage: python aggregate_stats.py /path/to/output-dir")
        sys.exit(1)

    output_dir = Path(sys.argv[1])
    slurm_out_dir = output_dir / "slurm_out"

    if not slurm_out_dir.exists():
        print(f"Error: {slurm_out_dir} does not exist")
        sys.exit(1)

    all_rows = []
    for out_file in sorted(slurm_out_dir.glob("*.out")):
        rows = parse_out_file(out_file)
        all_rows.extend(rows)
        print(f"  {out_file.name}: {len(rows)} stat rows")

    if not all_rows:
        print("No stats found.")
        sys.exit(1)

    # Sort by scene, split, step
    all_rows.sort(key=lambda r: (r["scene"] or "", r["split"] or "", r["step"] or 0))

    csv_path = output_dir / "stats.csv"
    fieldnames = ["scene", "split", "step", "psnr", "ssim", "lpips",
                  "ellipse_time", "num_GS", "mem_gb"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} rows to: {csv_path}")


if __name__ == "__main__":
    main()
