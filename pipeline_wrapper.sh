#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_SCRIPT="$SCRIPT_DIR/pipeline.sh"

DATA_ROOT="${DATA_ROOT:-../../siga26/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-../../siga26/output}"

# scene_name|output_scene|workflow_config
SCENE_CONFIGS=(
  # "figurines|figurines|configs/iterative_inpaint/workflow_figurines.json"
  # "bear|bear|configs/iterative_inpaint/workflow_bear.json"
  # "bonsai|bonsai|configs/iterative_inpaint/workflow_bonsai.json"
  "scene_1_colmap|scene_1_colmap|configs/iterative_inpaint/workflow_scene_1.json"
  "scene_5_colmap|scene_5_colmap|configs/iterative_inpaint/workflow_scene_5.json"
  "scene_6_colmap|scene_6_colmap|configs/iterative_inpaint/workflow_scene_6.json"
  "fruits|fruits|configs/iterative_inpaint/workflow_fruits.json"
)

SELECTED_SCENES=()
CUSTOM_RUNS=()
PASSTHROUGH_ARGS=()
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash pipeline_wrapper.sh [--scene NAME ...] [pipeline.sh options...]
  bash pipeline_wrapper.sh --run SOURCE_PATH MODEL_PATH WORKFLOW_CONFIG [pipeline.sh options...]

Behavior:
  - By default, runs all predefined scenes in this file.
  - Repeat --scene to run only selected predefined scenes.
  - Repeat --run to append explicit SOURCE_PATH, MODEL_PATH, WORKFLOW_CONFIG triples.
  - Any other arguments are forwarded to pipeline.sh and can override defaults.

Examples:
  bash pipeline_wrapper.sh
  bash pipeline_wrapper.sh --scene figurines --storage_mode minimal --overwrite
  bash pipeline_wrapper.sh --scene figurines --skip_train --skip_init --start_round 1
  bash pipeline_wrapper.sh \
    --run ../../siga26/data/figurines ../../siga26/output/figurines/3dgic configs/iterative_inpaint/workflow_figurines.json \
    --storage_mode lite --overwrite

Environment:
  DATA_ROOT     Scene root. Default: ../../siga26/data
  OUTPUT_ROOT   Output root. Default: ../../siga26/output
  PYTHON_BIN    Forwarded through pipeline.sh
  TORCH_HOME    Forwarded through pipeline.sh
EOF
}

fail() {
  echo "Error: $*" >&2
  exit 1
}

find_scene_config() {
  local scene_name="$1"
  local config=""
  for config in "${SCENE_CONFIGS[@]}"; do
    if [[ "${config%%|*}" == "$scene_name" ]]; then
      printf '%s\n' "$config"
      return 0
    fi
  done
  return 1
}

list_scene_names() {
  local config=""
  local names=()
  local IFS=' '
  for config in "${SCENE_CONFIGS[@]}"; do
    names+=("${config%%|*}")
  done
  printf '%s\n' "${names[*]}"
}

print_run_header() {
  local source_path="$1"
  local model_path="$2"
  local workflow_config="$3"

  echo "============================================================"
  echo "source_path     : $source_path"
  echo "model_path      : $model_path"
  echo "workflow_config : $workflow_config"
  echo "============================================================"
}

run_pipeline() {
  local source_path="$1"
  local model_path="$2"
  local workflow_config="$3"
  local pipeline_args=(
    bash "$PIPELINE_SCRIPT"
    -s "$source_path"
    -m "$model_path"
    --workflow_config "$workflow_config"
    "${PASSTHROUGH_ARGS[@]}"
  )

  [[ -d "$source_path" ]] || fail "source path does not exist: $source_path"
  [[ -d "$source_path/object_mask" ]] || fail "source path must contain object_mask/: $source_path/object_mask"
  [[ -f "$workflow_config" ]] || fail "workflow config not found: $workflow_config"

  print_run_header "$source_path" "$model_path" "$workflow_config"

  if (( DRY_RUN )); then
    printf 'Dry run command:'
    printf ' %q' "${pipeline_args[@]}"
    printf '\n'
    return 0
  fi

  "${pipeline_args[@]}"
}

run_predefined_scene() {
  local scene_name="$1"
  local config=""
  local source_scene=""
  local output_scene=""
  local workflow_config=""

  config="$(find_scene_config "$scene_name")" || {
    echo "Unknown scene: $scene_name" >&2
    echo "Available scenes: $(list_scene_names)" >&2
    exit 1
  }

  IFS='|' read -r source_scene output_scene workflow_config <<< "$config"
  run_pipeline "$DATA_ROOT/$source_scene" "$OUTPUT_ROOT/$output_scene/3dgic" "$workflow_config"
}

run_custom_entry() {
  local entry="$1"
  local source_path=""
  local model_path=""
  local workflow_config=""

  IFS='|' read -r source_path model_path workflow_config <<< "$entry"
  run_pipeline "$source_path" "$model_path" "$workflow_config"
}

if [[ ! -f "$PIPELINE_SCRIPT" ]]; then
  fail "missing pipeline script: $PIPELINE_SCRIPT"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene)
      [[ $# -ge 2 ]] || fail "missing value for --scene"
      SELECTED_SCENES+=("$2")
      shift 2
      ;;
    --run)
      [[ $# -ge 4 ]] || fail "--run requires SOURCE_PATH MODEL_PATH WORKFLOW_CONFIG"
      CUSTOM_RUNS+=("$2|$3|$4")
      shift 4
      ;;
    --data_root)
      [[ $# -ge 2 ]] || fail "missing value for --data_root"
      DATA_ROOT="$2"
      shift 2
      ;;
    --output_root)
      [[ $# -ge 2 ]] || fail "missing value for --output_root"
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --list-scenes)
      list_scene_names
      exit 0
      ;;
    --dry_run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
  esac
done

cd "$SCRIPT_DIR"

if [[ ${#SELECTED_SCENES[@]} -eq 0 && ${#CUSTOM_RUNS[@]} -eq 0 ]]; then
  for config in "${SCENE_CONFIGS[@]}"; do
    run_predefined_scene "${config%%|*}"
  done
else
  for scene_name in "${SELECTED_SCENES[@]}"; do
    run_predefined_scene "$scene_name"
  done
  for entry in "${CUSTOM_RUNS[@]}"; do
    run_custom_entry "$entry"
  done
fi


# 直接加 --use_depth_loss 就行了，pipeline.sh 里会覆盖默认值，没必要在这里加一个环境变量了