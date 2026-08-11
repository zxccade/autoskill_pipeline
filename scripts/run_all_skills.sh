#!/usr/bin/env bash
# =============================================================================
# Run all skills × all benchmarks (6 skills × 5 benchmarks = 30 runs)
# Skips already-completed runs (checkpoint resume).
#
# Usage: bash run_all_skills.sh [output_dir]
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/run_benchmark_skill.sh"
OUT_DIR="${1:-./outputs}"

SKILLS=(
    uniform_128_direct
    siglip_mmr_diverse
    clip_spatial_cooccur
    clip_count_topk
    siglip_ocr_text_aware
    clip_mmr_diverse
)

BENCHMARKS=(videomme mlvu_dev mlvu_test lvbench longvideobench)

TOTAL=$((${#SKILLS[@]} * ${#BENCHMARKS[@]}))
echo "=========================================="
echo " AutoSkill — All Skills × All Benchmarks"
echo " ${#SKILLS[@]} skills × ${#BENCHMARKS[@]} benchmarks = $TOTAL runs"
echo "=========================================="

COUNT=0; SKIPPED=0; FAILED=0
for BENCH in "${BENCHMARKS[@]}"; do
    echo ""
    echo "========== BENCHMARK: $BENCH =========="
    for SKILL in "${SKILLS[@]}"; do
        COUNT=$((COUNT + 1))
        RUN_LOG="$OUT_DIR/${BENCH}_$SKILL/run.log"

        # Checkpoint: skip if already done
        if [ -f "$RUN_LOG" ] && tail -1 "$RUN_LOG" 2>/dev/null | grep -q "DONE skill=$SKILL bench=$BENCH"; then
            echo "[$COUNT/$TOTAL] SKIP (done): $SKILL on $BENCH"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        echo "[$COUNT/$TOTAL] skill=$SKILL on $BENCH"
        bash "$RUNNER" "$SKILL" "$BENCH" "$OUT_DIR" || {
            echo "!!! FAILED [$COUNT/$TOTAL] $SKILL on $BENCH — continuing"
            FAILED=$((FAILED + 1))
        }
    done
done

echo ""
echo "ALL DONE ($COUNT/$TOTAL) — done=$((COUNT - SKIPPED - FAILED)) skipped=$SKIPPED failed=$FAILED"
