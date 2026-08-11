#!/usr/bin/env bash
# =============================================================================
# Run single skill × single benchmark using lmms-eval
# Auto-detects AMD ROCm / NVIDIA CUDA, GPU count, and venv.
#
# Usage: bash run_benchmark_skill.sh <skill_name> <benchmark> [output_dir]
#   skill_name : uniform_128_direct, siglip_mmr_diverse, clip_spatial_cooccur, ...
#   benchmark  : videomme | mlvu_dev | mlvu_test | lvbench | longvideobench
#   output_dir : output directory (default: ./outputs)
# =============================================================================
set -euo pipefail

SKILL="${1:?Usage: $0 <skill_name> <benchmark> [output_dir]}"
BENCH="${2:?Usage: $0 <skill_name> <benchmark> [output_dir]}"
OUT_DIR="${3:-./outputs}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LMMS_DIR="${LMMS_DIR:-$SCRIPT_DIR/../lmms-eval}"
MODEL_PATH="${MODEL_PATH:-/path/to/Qwen2.5-VL-7B-Instruct}"
MAX_FRAMES="${MAX_FRAMES:-128}"

LOG_DIR="$OUT_DIR/${BENCH}_$SKILL"
mkdir -p "$LOG_DIR"

# ── Environment ──────────────────────────────────────────────────────────────
export AUTO_SKILL_LMMS_DIR="$LMMS_DIR"
export SKILL_LEARNING_PROJECT_DIR="$LMMS_DIR"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export FORCE_QWENVL_VIDEO_READER=decord
export TOKENIZERS_PARALLELISM=false

# ── Auto-detect GPU backend ──────────────────────────────────────────────────
TORCH_BACKEND=$(python -c "import torch; print('rocm' if getattr(torch.version, 'hip', None) else 'cuda')" 2>/dev/null || echo "cuda")

if [ "$TORCH_BACKEND" = "rocm" ]; then
    export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
    GPU_LABEL="AMD ROCm"
else
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    GPU_LABEL="NVIDIA CUDA"
fi

# Auto-detect GPU count
if [ -z "${N_GPUS:-}" ]; then
    if [ "$TORCH_BACKEND" = "cuda" ] && [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        N_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
    elif [ "$TORCH_BACKEND" = "rocm" ] && [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
        N_GPUS=$(echo "$ROCR_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
    else
        N_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 8)
    fi
fi
echo "Using $N_GPUS GPUs ($GPU_LABEL)"

cd "$LMMS_DIR"
export PYTHONPATH="$LMMS_DIR:${PYTHONPATH:-}"

# ── Task mapping ─────────────────────────────────────────────────────────────
case "$BENCH" in
    videomme)        TASKS="videomme" ;;
    mlvu_dev)        TASKS="mlvu_dev" ;;
    mlvu_test)       TASKS="mlvu_test" ;;
    lvbench)         TASKS="lvbench" ;;
    longvideobench)  TASKS="longvideobench_val_v" ;;
    *) echo "ERROR: Unknown benchmark '$BENCH'" >&2; exit 1 ;;
esac

COMMON_ARGS="pretrained=${MODEL_PATH},skill_name=${SKILL},device_map=auto,attn_implementation=sdpa,max_frames=${MAX_FRAMES}"

# Random free port for accelerate
get_free_port() {
    while port=$(shuf -i 20000-29999 -n 1); ss -tln 2>/dev/null | awk '{print $4}' | grep -qE ":${port}$"; do :; done
    echo "$port"
}
MASTER_PORT="$(get_free_port)"

echo "[$(date '+%F %T')] START skill=$SKILL bench=$BENCH gpus=$N_GPUS port=$MASTER_PORT"

python -m accelerate.commands.launch \
    --num_processes "$N_GPUS" \
    --num_machines 1 \
    --machine_rank 0 \
    --main_process_port "$MASTER_PORT" \
    -m lmms_eval \
        --model qwen2_5_vl_skill \
        --model_args "$COMMON_ARGS" \
        --tasks "$TASKS" \
        --batch_size 1 \
        --output_path "$LOG_DIR" \
        --log_samples \
        --log_samples_suffix "$BENCH"

echo "[$(date '+%F %T')] DONE skill=$SKILL bench=$BENCH"
