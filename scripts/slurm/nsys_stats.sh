#!/bin/bash
# Usage: ./nsys_stats.sh /path/to/nsys-rep-folder [output.txt]

INPUT_DIR="${1:-.}"
OUTPUT_TXT="${2:-nsys_stats_summary.txt}"

> "$OUTPUT_TXT"  # clear/create output file

for REP_FILE in "$INPUT_DIR"/*.nsys-rep; do
    [ -f "$REP_FILE" ] || { echo "No .nsys-rep files found."; exit 1; }

    BASENAME=$(basename "$REP_FILE")
    echo "Processing: $BASENAME"

    {
        echo "========================================"
        echo "FILE: $BASENAME"
        echo "========================================"
        nsys stats "$REP_FILE"
        echo ""
    } >> "$OUTPUT_TXT"

done

echo "Done. Results written to: $OUTPUT_TXT"