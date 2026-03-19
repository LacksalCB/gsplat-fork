#!/bin/bash
# Usage: ./nsys_stats.sh /path/to/nsys-rep-folder [output.txt]

INPUT_DIR="${1:-.}"
OUTPUT_TXT="${2:-nsys_stats_summary.txt}"
SLURM_OUT_DIR="$(dirname $INPUT_DIR)/slurm_out"

> "$OUTPUT_TXT"  # clear/create output file

for REP_FILE in "$INPUT_DIR"/*.nsys-rep; do
    [ -f "$REP_FILE" ] || { echo "No .nsys-rep files found."; exit 1; }

    BASENAME=$(basename "$REP_FILE")
    SCENE=$(echo "$BASENAME" | sed 's/gsplat_profile_\(.*\)\.nsys-rep/\1/')
    echo "Processing: $BASENAME (scene: $SCENE)"

    # Find the .out file whose content mentions this scene
    RUNTIME_STR="Runtime: N/A (no matching .out file found)"
    if [ -d "$SLURM_OUT_DIR" ]; then
        OUT_FILE=$(grep -rl "Running $SCENE" "$SLURM_OUT_DIR" 2>/dev/null | head -1)
        if [ -n "$OUT_FILE" ]; then
            START_LINE=$(head -1 "$OUT_FILE")
            START_EPOCH=$(date -d "$START_LINE" +%s 2>/dev/null)
            END_EPOCH=$(stat --format="%Y" "$OUT_FILE")
            if [ -n "$START_EPOCH" ] && [ "$START_EPOCH" -gt 0 ]; then
                DURATION=$((END_EPOCH - START_EPOCH))
                HOURS=$((DURATION / 3600))
                MINS=$(( (DURATION % 3600) / 60 ))
                SECS=$((DURATION % 60))
                START_STR=$(date -d "@$START_EPOCH" "+%Y-%m-%d %H:%M:%S")
                END_STR=$(date -d "@$END_EPOCH" "+%Y-%m-%d %H:%M:%S")
                RUNTIME_STR=$(printf "Runtime: %02d:%02d:%02d  (start: %s  end: %s)" \
                    $HOURS $MINS $SECS "$START_STR" "$END_STR")
            fi
        fi
    fi

    {
        echo "========================================"
        echo "FILE: $BASENAME"
        echo "$RUNTIME_STR"
        echo "========================================"
        nsys stats "$REP_FILE"
        echo ""
    } >> "$INPUT_DIR/$OUTPUT_TXT"

done

echo "Done. Results written to: "$INPUT_DIR/$OUTPUT_TXT""
