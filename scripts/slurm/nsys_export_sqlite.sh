#!/bin/bash
# Export all .nsys-rep files in a directory to .sqlite in parallel.
#
# Usage:
#   ./nsys_export_sqlite.sh <results_dir>
#
# Example:
#   ./nsys_export_sqlite.sh outputs/2026-04-10_21-39-15_vram_sweep/results

DIR="${1:-.}"

if [ ! -d "$DIR" ]; then
    echo "Error: directory not found: $DIR"
    exit 1
fi

REPS=("$DIR"/*.nsys-rep)
if [ ! -f "${REPS[0]}" ]; then
    echo "No .nsys-rep files found in $DIR"
    exit 1
fi

echo "Exporting ${#REPS[@]} .nsys-rep files in: $DIR"

for REP in "${REPS[@]}"; do
    echo "  Queuing: $(basename "$REP")"
    nsys export --type sqlite --force-overwrite true "$REP" &
done

wait
echo "All exports done."
