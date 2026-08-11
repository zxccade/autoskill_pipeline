#!/usr/bin/env bash
# =============================================================================
# AutoSkill Pipeline — Qwen2.5-VL-7B 完整 5 阶段运行脚本
#
# 用法:
#   bash run_pipeline.sh                          # 跑全部阶段
#   bash run_pipeline.sh --stage 3                # 只跑阶段 3
#   bash run_pipeline.sh --stages 3,4,5           # 跑阶段 3-5
#
# 环境变量 (在调用前设置或在此修改):
#   MODEL_PATH        Qwen2.5-VL-7B 模型路径
#   BENCH_DIR         benchmark per-sample 结果目录
#   N1800_RESULTS     N1800 results.json
#   SUPP_RESULTS       supplement results.json
#   N1800_META        N1800 metadata JSON
#   SUPP_META         supplement metadata JSON
#   TRAIN_JSON        原始训练集 JSON
#   BENCH_QUERIES     benchmark query 列表 JSON
#   TRAIN_SAMPLES     已选好的 1500 训练样本 JSONL (含 skill_results)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 默认配置 (请根据实际路径修改) ──
MODEL_PATH="${MODEL_PATH:-/path/to/Qwen2.5-VL-7B-Instruct}"
BENCH_DIR="${BENCH_DIR:-/path/to/qwen25vl_skill_260518_top10/raw_results/benchmarks}"
N1800_RESULTS="${N1800_RESULTS:-/path/to/N1800_amd8_*/results.json}"
SUPP_RESULTS="${SUPP_RESULTS:-/path/to/supplement_amd8_*/results.json}"
N1800_META="${N1800_META:-/path/to/eval_router_N1800_metadata.json}"
SUPP_META="${SUPP_META:-/path/to/eval_supplement_eval_custom.json}"
TRAIN_JSON="${TRAIN_JSON:-/path/to/eval_3k_augmented.json}"
BENCH_QUERIES="${BENCH_QUERIES:-/path/to/classification.json}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-/path/to/n1800_supp_6cand_1500.jsonl}"

SKILLS="siglip_mmr_diverse,clip_spatial_cooccur,clip_count_topk,siglip_ocr_text_aware,clip_mmr_diverse,uniform_128_direct"
BENCHMARKS="mlvu,mlvu_test,longvideobench,videomme,lvbench"

OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/outputs}"
mkdir -p "$OUT_DIR"

# ── 解析参数 ──
STAGE=""
STAGES=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)  STAGE="$2"; shift 2;;
        --stages) STAGES="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

run_stage() {
    local n="$1"
    case "$n" in
        1) echo "========== Stage 1: Skill Discovery =========="
           python3 "$SCRIPT_DIR/stage1_skill_discovery.py" \
               --n1800-results "$N1800_RESULTS" \
               --supp-results "$SUPP_RESULTS" \
               --n1800-meta "$N1800_META" \
               --supp-meta "$SUPP_META" \
               --output "$OUT_DIR/top5_skills.json"
           ;;
        2) echo "========== Stage 2: Rewrite Training Set =========="
           python3 "$SCRIPT_DIR/stage2_rewrite_trainset.py" \
               --train-json "$TRAIN_JSON" \
               --bench-queries "$BENCH_QUERIES" \
               --model-path "$MODEL_PATH" \
               --output "$OUT_DIR/train_rewritten.json"
           ;;
        3) echo "========== Stage 3: Build Router Table =========="
           python3 "$SCRIPT_DIR/stage3_build_router.py" \
               --train-samples "$TRAIN_SAMPLES" \
               --top5-skills "$OUT_DIR/top5_skills.json" \
               --output "$OUT_DIR/router_table.json"
           ;;
        4) echo "========== Stage 4: Classify Benchmark =========="
           python3 "$SCRIPT_DIR/stage4_classify_benchmark.py" \
               --bench-dir "$BENCH_DIR" \
               --skills "$SKILLS" \
               --benchmarks "$BENCHMARKS" \
               --model-path "$MODEL_PATH" \
               --output "$OUT_DIR/cls_pred_v3.json"
           ;;
        5) echo "========== Stage 5: Evaluate =========="
           python3 "$SCRIPT_DIR/stage5_evaluate.py" \
               --bench-dir "$BENCH_DIR" \
               --skills "$SKILLS" \
               --router-table "$OUT_DIR/router_table.json" \
               --cls-pred "$OUT_DIR/cls_pred_v3.json" \
               --benchmarks "$BENCHMARKS"
           ;;
    esac
}

if [[ -n "$STAGE" ]]; then
    run_stage "$STAGE"
elif [[ -n "$STAGES" ]]; then
    IFS=',' read -ra STAGE_LIST <<< "$STAGES"
    for s in "${STAGE_LIST[@]}"; do
        run_stage "$(echo "$s" | tr -d ' ')"
    done
else
    for s in 1 2 3 4 5; do
        run_stage "$s"
    done
fi

echo ""
echo "Pipeline complete. Outputs in: $OUT_DIR/"
