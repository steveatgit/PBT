#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/compare_PBT.sh ADAPTER_CHECKPOINT [PRETRAINED_CHECKPOINT]
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
adapter_checkpoint="${1:-${PBT_ADAPTER_CHECKPOINT:-}}"
if [[ -z "$adapter_checkpoint" || $# -gt 2 ]]; then
  printf 'Usage: bash scripts/compare_PBT.sh ADAPTER_CHECKPOINT [PRETRAINED_CHECKPOINT]\n' >&2
  exit 1
fi
pretrained_checkpoint="${2:-${PBT_CHECKPOINT:-$repo_root/checkpoints/PBT_pretrained/PBTs/PBT_10_Llama_1_le80_bs128_lr2.5e-05_dm128_nh8_el2_dl10_df128_mdf64_lradjconstant_MIX_large_guideFalse_LBFalse_lossMSE_wd0.01_wlFalse_dr0.05_gdff512_E5_GE5_K-1_SFalse_augFalse_augW1.0_tem1.0_wDGFalse_dsr0.75_we0_ffsTrue_seed42-100}}"
dataset="${EVAL_DATASET:-CALB42}"
cycle_min="${EVAL_CYCLE_MIN:-10}"
cycle_max="${EVAL_CYCLE_MAX:-10}"
batch_size="${BATCH_SIZE:-4}"
root_path="${PBT_DATASET_ROOT:-$repo_root/dataset}"
output_dir="${PBT_COMPARE_OUTPUT:-$repo_root/results/compare/$(date +%Y%m%d_%H%M%S)_$$}"
export PYTHONUNBUFFERED=1

# Reject mismatched checkpoints before spending time evaluating.
python - "$pretrained_checkpoint" "$adapter_checkpoint" "$dataset" <<'PY'
import json
import sys
from pathlib import Path

configs = []
for folder in sys.argv[1:3]:
    path = Path(folder)
    for name in ('args.json', 'model.safetensors', 'label_scaler'):
        if not (path / name).is_file():
            raise SystemExit(f'Missing checkpoint file: {path / name}')
    configs.append(json.loads((path / 'args.json').read_text()))
base, adapter = configs
if base.get('model') != 'PBT' or adapter.get('model') != 'PBT':
    raise SystemExit('Both checkpoints must be PBT models.')
if base.get('finetune_method'):
    raise SystemExit('The baseline must be a pretrained checkpoint.')
if adapter.get('finetune_method') not in ('AT', 'AT_reverse', 'AT_nCP'):
    raise SystemExit('The comparison checkpoint must use adapter tuning.')
if base['seed'] != adapter['seed']:
    raise SystemExit('Checkpoint seeds differ; use the base checkpoint for this adapter run.')
if adapter['dataset'] != sys.argv[3]:
    raise SystemExit(f"Adapter dataset is {adapter['dataset']}; set EVAL_DATASET accordingly.")
PY

if [[ -e "$output_dir" ]]; then
  printf 'Comparison directory already exists; choose a new PBT_COMPARE_OUTPUT: %s\n' "$output_dir" >&2
  exit 1
fi
mkdir -p "$output_dir"
printf 'Dataset: %s | Input cycles: %s-%s\nResults: %s\n' "$dataset" "$cycle_min" "$cycle_max" "$output_dir"

compare_model() {
  local checkpoint="$1" variant="$2"
  mkdir -p "$output_dir/$variant"
  accelerate launch --mixed_precision no --num_processes 1 \
    --num_machines 1 --dynamo_backend no evaluate_model.py \
    --args_path "${checkpoint%/}/" --root_path "$root_path" \
    --batch_size "$batch_size" --num_workers 0 \
    --eval_dataset "$dataset" \
    --eval_cycle_min "$cycle_min" --eval_cycle_max "$cycle_max" \
    --results_dir "$output_dir/$variant" \
    --metrics_output "$output_dir/$variant/metrics.txt" \
    2>&1 | tee "$output_dir/$variant/evaluate.log"
}

compare_model "$pretrained_checkpoint" pretrained
compare_model "$adapter_checkpoint" adapter

python - "$output_dir" "$dataset" "$pretrained_checkpoint" "$adapter_checkpoint" "$cycle_min" "$cycle_max" <<'PY' | tee "$output_dir/summary.txt"
import csv
import json
import math
import sys
from pathlib import Path

output = Path(sys.argv[1])
dataset = sys.argv[2]

def read_result(variant):
    files = list((output / variant).glob('PBT_*.json'))
    if len(files) != 1:
        raise SystemExit(f'Expected one evaluation JSON for {variant}, found {len(files)}.')
    return json.loads(files[0].read_text())[dataset]

base, adapter = (read_result(variant) for variant in ('pretrained', 'adapter'))
for field in ('Useable_cycle_number', 'total_seen_unseen_ids', 'domain_ids'):
    if base[field] != adapter[field]:
        raise SystemExit(f'Evaluation samples differ in {field}; comparison aborted.')
if len(base['total_references']) != len(adapter['total_references']) or not all(
    math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-3)
    for a, b in zip(base['total_references'], adapter['total_references'])
):
    raise SystemExit('Test labels differ; comparison aborted.')

def read_metrics(variant):
    return dict(line.split(': ', 1) for line in (output / variant / 'metrics.txt').read_text().splitlines())

before, after = (read_metrics(variant) for variant in ('pretrained', 'adapter'))
metrics = (
    ('cell_level_mape', 'Cell MAPE'),
    ('aging_condition_level_mape', 'Condition MAPE'),
    ('seen_aging_condition_level_mape', 'Seen condition MAPE'),
    ('unseen_aging_condition_level_mape', 'Unseen condition MAPE'),
)
rows = []
print(f'Dataset: {dataset} | Input cycles: {sys.argv[5]}-{sys.argv[6]} | Test samples: {len(base["total_references"])}')
print(f'Pretrained checkpoint: {sys.argv[3]}')
print(f'Adapter checkpoint: {sys.argv[4]}')
print('MAPE shown as percentages; lower is better. Positive improvement means lower error.\n')
print(f'{"Metric":<25} {"Pretrained":>12} {"Adapter":>12} {"Improvement":>14}')
for key, label in metrics:
    old = None if before[key] == 'NA' else float(before[key])
    new = None if after[key] == 'NA' else float(after[key])
    improvement = (old - new) / old * 100 if old is not None and new is not None and old > 0 else None
    fmt = lambda value: 'NA' if value is None else f'{value:.2f}%'
    print(f'{label:<25} {fmt(None if old is None else old * 100):>12} {fmt(None if new is None else new * 100):>12} {fmt(improvement):>14}')
    rows.append((key, old, new, improvement))
with (output / 'comparison.csv').open('w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(('metric', 'pretrained_mape_fraction', 'adapter_mape_fraction', 'relative_improvement_percent'))
    writer.writerows(rows)
PY
