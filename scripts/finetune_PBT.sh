#!/usr/bin/env bash
set -euo pipefail

# Run from any directory. Defaults target the verified CALB42/seed42 checkpoint.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
args_path="${PBT_CHECKPOINT:-$repo_root/checkpoints/PBT_pretrained/PBTs/PBT_10_Llama_1_le80_bs128_lr2.5e-05_dm128_nh8_el2_dl10_df128_mdf64_lradjconstant_MIX_large_guideFalse_LBFalse_lossMSE_wd0.01_wlFalse_dr0.05_gdff512_E5_GE5_K-1_SFalse_augFalse_augW1.0_tem1.0_wDGFalse_dsr0.75_we0_ffsTrue_seed42-100}"
args_path="${args_path%/}/"
root_path="${PBT_DATASET_ROOT:-$repo_root/dataset}"
checkpoints="${PBT_FINETUNE_OUTPUT:-$repo_root/checkpoints/PBT_adapter}"
finetune_dataset="${FINETUNE_DATASET:-CALB42}" # Must match checkpoint seed: CALB42 for seed42.
batch_size="${BATCH_SIZE:-4}"
train_epochs="${TRAIN_EPOCHS:-300}"
least_epochs="${LEAST_EPOCHS:-50}"
patience="${PATIENCE:-30}"
learning_rate="${LEARNING_RATE:-0.000025}"
adapter_size="${ADAPTER_SIZE:-16}"
adapter_layers="${ADAPTER_LAYERS:--1}" # All encoder/decoder layers, plus CyclePatch adapter.
seq_len="${SEQ_LEN:-1}"
early_cycle_threshold="${EARLY_CYCLE_THRESHOLD:-100}"
# The trainer clears an existing run directory; use a unique name each time.
comment="${RUN_NAME:-${finetune_dataset}_AT_$(date +%Y%m%d_%H%M%S)_$$}"
# Track against the local W&B server by default. Environment variables can
# override the server address or select offline/disabled mode.
export WANDB_BASE_URL="${WANDB_BASE_URL:-http://localhost:8080}"
export WANDB_MODE="${WANDB_MODE:-online}"

export PYTHONUNBUFFERED=1
log_dir="${PBT_LOG_DIR:-$repo_root/logs/finetune}"
mkdir -p "$log_dir"
log_file="$log_dir/$comment.log"

run_finetune() {
  printf 'Run: %s\nLog: %s\nCheckpoint root: %s\n' "$comment" "$log_file" "$checkpoints"
  printf 'W&B mode: %s | Server: %s\n' "$WANDB_MODE" "$WANDB_BASE_URL"
  for required in args.json model.safetensors label_scaler; do
    if [[ ! -f "$args_path$required" ]]; then
      printf 'Missing checkpoint file: %s\n' "$args_path$required" >&2
      exit 1
    fi
  done
  if [[ -e "$checkpoints/$comment" ]]; then
    printf 'Run directory already exists; choose a different RUN_NAME: %s\n' "$checkpoints/$comment" >&2
    exit 1
  fi

  # macOS/MPS: full precision, one process, no data-loader subprocesses.
  accelerate launch --mixed_precision no --num_processes 1 \
    --num_machines 1 --dynamo_backend no finetune_model.py \
    --model PBT --task_name battery_life_prediction --is_training 1 \
    --args_path "$args_path" --root_path "$root_path" \
    --checkpoints "$checkpoints" --model_comment "$comment" \
    --finetune_dataset "$finetune_dataset" --finetune_method AT \
    --adapter_size "$adapter_size" --adapter_layers "$adapter_layers" \
    --batch_size "$batch_size" --num_workers 0 --num_process 1 \
    --learning_rate "$learning_rate" --train_epochs "$train_epochs" \
    --least_epochs "$least_epochs" --patience "$patience" \
    --seq_len "$seq_len" --early_cycle_threshold "$early_cycle_threshold" \
    --dropout 0.0 --wd 0.0 --loss MSE --lradj constant \
    --warm_up_epoches 0 --topK -1 --itr 1 --accumulation_steps 1
}

# Capture stdout and stderr while streaming them to the terminal. With pipefail,
# a failed trainer remains a failed script even when tee succeeds.
run_finetune 2>&1 | tee -a "$log_file"
