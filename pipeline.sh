#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
read -r -a PYTHON_CMD <<< "$PYTHON_BIN"

export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_HOME="${TORCH_HOME:-${SCRIPT_DIR}/../pretrained_models/torch}"

SOURCE_PATH=""
MODEL_PATH=""
WORKFLOW_CONFIG=""
TARGET_IDS=()
TARGET_GROUPS=()

TRAIN_CONFIG="${SCRIPT_DIR}/configs/gaussian_dataset/train.json"
REMOVAL_CONFIG_TEMPLATE=""
INPAINT_CONFIG_TEMPLATE=""

ITERATIONS=30000
SAVE_INTERVAL=5000
CHECKPOINT_INTERVAL=5000
TEST_INTERVAL=2500
DENSIFICATION_INTERVAL=500
LAMBDA_NORMAL_RENDER_DEPTH=0.01
LAMBDA_MASK_ENTROPY=0.1
SAVE_TRAINING_VIS=1
SAVE_TRAINING_VIS_ITERATION=1000
EVAL=1
USE_DEPTH_LOSS=0
DEPTHS=""
DEPTH_SCALE=0.0
DEPTH_L1_WEIGHT_INIT=1.0
DEPTH_L1_WEIGHT_FINAL=0.01

TOP_K_REF_VIEWS=""
SIMPLE_LAMA_DEVICE="${SIMPLE_LAMA_DEVICE:-}"
MASK_THRESHOLD=""
MASK_DILATION=""
INTERSECT_TOP_M=""
INTERSECT_CACHE_SIZE=""
BACKGROUND_ID=""
FINETUNE_ITERATION=""
START_ROUND=0
STORAGE_MODE=full
RENDER_INTERMEDIATE=0

SKIP_TRAIN=0
SKIP_INIT=0
OVERWRITE=0
ALLOW_EXISTING_TRAIN_OUTPUT=0

usage() {
  cat <<'EOF'
Usage:
  bash pipeline.sh \
    -s <source_path> \
    -m <model_path> \
    --workflow_config <workflow_json> \
    [options]

End-to-end stages:
  1. Train the initial 3DGIC/GaussianGrouping-style 3DGS model with object features.
  2. Initialize <model_path>/iterative_3dgic/workflow.json.
  3. Run each configured target group as one iterative 3DGIC inpaint round.

Required arguments:
  -s, --source_path                  Dataset root. It must contain images and object_mask/.
  -m, --model_path                   Output model root used by training and iterative inpaint.
  --workflow_config, --config_file   Workflow JSON with a non-empty rounds array.

Round shorthand:
  --target_groups <g1> [g2 ...]      Terminal groups, one token per round. Example: 34,57 81 90,91.
  --target_ids <id1> [id2 ...]       Legacy shorthand: one id per round. Example: 34 57 81.
                                      Either shorthand may be used instead of --workflow_config for quick runs.

Training options:
  --skip_train                       Do not run train.py; require an existing trained model under --model_path.
  --allow_existing_train_output      Permit train.py to write into an existing model_path/point_cloud.
  --train_config <path>              Default: configs/gaussian_dataset/train.json.
  --iterations <int>                 Default: 40000.
  --save_interval <int>              Default: 5000.
  --checkpoint_interval <int>        Default: 5000.
  --test_interval <int>              Default: 2500.
  --densification_interval <int>     Default: 500.
  --lambda_normal_render_depth <v>   Default: 0.01.
  --lambda_mask_entropy <v>          Default: 0.1.
  --use_depth_loss                   Enable raw .npy inverse-depth supervision during base training.
  --depths <path>                    Raw .npy depth folder. Default when enabled: <source_path>/depth.
  --depth_scale <v>                  Raw-depth to COLMAP/3DGIC scale. Default: 0.0, estimate from COLMAP tracks.
  --depth_l1_weight_init <v>         Initial inverse-depth loss weight. Default: 1.0.
  --depth_l1_weight_final <v>        Final inverse-depth loss weight. Default: 0.01.
  --no_eval                          Do not pass --eval to train.py.
  --no_save_training_vis             Do not save training visualization grids.
  --save_training_vis_iteration <n>  Default: 1000.

Iterative inpaint options:
  --skip_init                        Reuse an existing iterative_3dgic/workflow.json.
  --overwrite                        Replace workflow on init and overwrite executed round work dirs.
  --start_round <int>                Resume from this round index. Default: 0.
  --removal_config_template <path>   Override config/default removal template.
  --inpaint_config_template <path>   Override config/default inpaint template.
  --top_k_ref_views <int>            Number of reference views selected by mask area. Default: 3.
  --simple_lama_device <device>      Device for SimpleLaMa. Default: cuda.
  --mask_threshold <int>             Reference-mask threshold. Default: 0.
  --mask_dilation <int>              Reference-mask dilation radius. Default: 0.
  --intersect_top_m <int>            Exact intersect masks are computed only for top-M ordinary masks. Default: 20; 0 means all.
  --intersect_cache_size <int>       Source-view intersect cache size. Default: 256; 0 disables cache.
  --background_id <int>              Label used to suppress completed ids in later rounds. Default: 0.
  --finetune_iteration <int>         Override object-inpaint finetune_iteration from the template.
  --storage_mode <full|lite|minimal> Output retention mode. Default: full.
                                      lite skips non-final inpaint renders and prunes heavy point-cloud intermediates.
                                      minimal also removes round data_work/scene_in after scene_out is saved.
  --render_intermediate              Render inpaint outputs for non-final rounds. Default: disabled for lite/minimal.

Environment:
  PYTHON_BIN                         Python command. Default: python. Example: "conda run -n 3dgic python".
  SIMPLE_LAMA_DEVICE                 Device override for SimpleLaMa if --simple_lama_device is not set.
  TORCH_HOME                         Default: ../pretrained_models/torch.

Example:
  bash pipeline.sh \
    -s ./data/bear \
    -m ./output/NeRF_Syn/bear/3dgs \
    --workflow_config configs/iterative_inpaint/workflow_template.json \
    --overwrite

  bash pipeline.sh \
    -s ./data/bear \
    -m ./output/NeRF_Syn/bear/3dgs \
    --target_groups 34,57 81 90,91 \
    --storage_mode minimal \
    --overwrite
EOF
}

fail() {
  echo "Error: $*" >&2
  exit 1
}

write_time_consuming() {
  local end_time end_text elapsed hours minutes seconds elapsed_hms output_path
  end_time="$(date +%s)"
  end_text="$(date '+%F %T')"
  elapsed=$((end_time - PIPELINE_START_TIME))
  hours=$((elapsed / 3600))
  minutes=$(((elapsed % 3600) / 60))
  seconds=$((elapsed % 60))
  printf -v elapsed_hms "%02d:%02d:%02d" "$hours" "$minutes" "$seconds"
  mkdir -p "$MODEL_PATH"
  output_path="$MODEL_PATH/time_consuming.txt"
  {
    printf 'source_path: %s\n' "$SOURCE_PATH"
    printf 'model_path: %s\n' "$MODEL_PATH"
    printf 'start_time: %s\n' "$PIPELINE_START_TEXT"
    printf 'end_time: %s\n' "$end_text"
    printf 'elapsed_seconds: %d\n' "$elapsed"
    printf 'elapsed_hms: %s\n' "$elapsed_hms"
  } > "$output_path"
  echo "[$end_text] Total pipeline time: $elapsed_hms ($elapsed seconds). Wrote $output_path"
}

append_target_ids() {
  local raw="$1"
  local part
  local parts
  IFS=',' read -r -a parts <<< "$raw"
  for part in "${parts[@]}"; do
    [[ -n "$part" ]] && TARGET_IDS+=("$part")
  done
}

append_target_group() {
  local raw="$1"
  [[ -n "$raw" ]] && TARGET_GROUPS+=("$raw")
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--source_path)
      SOURCE_PATH="$2"
      shift 2
      ;;
    -m|--model_path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --workflow_config|--config_file)
      WORKFLOW_CONFIG="$2"
      shift 2
      ;;
    --target_groups)
      shift
      while [[ $# -gt 0 && "$1" != -* ]]; do
        append_target_group "$1"
        shift
      done
      ;;
    --target_ids)
      shift
      while [[ $# -gt 0 && "$1" != -* ]]; do
        append_target_ids "$1"
        shift
      done
      ;;
    --skip_train)
      SKIP_TRAIN=1
      shift
      ;;
    --allow_existing_train_output)
      ALLOW_EXISTING_TRAIN_OUTPUT=1
      shift
      ;;
    --train_config)
      TRAIN_CONFIG="$2"
      shift 2
      ;;
    --iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    --save_interval)
      SAVE_INTERVAL="$2"
      shift 2
      ;;
    --checkpoint_interval)
      CHECKPOINT_INTERVAL="$2"
      shift 2
      ;;
    --test_interval)
      TEST_INTERVAL="$2"
      shift 2
      ;;
    --densification_interval)
      DENSIFICATION_INTERVAL="$2"
      shift 2
      ;;
    --lambda_normal_render_depth)
      LAMBDA_NORMAL_RENDER_DEPTH="$2"
      shift 2
      ;;
    --lambda_mask_entropy)
      LAMBDA_MASK_ENTROPY="$2"
      shift 2
      ;;
    --use_depth_loss)
      USE_DEPTH_LOSS=1
      shift
      ;;
    --depths)
      DEPTHS="$2"
      shift 2
      ;;
    --depth_scale)
      DEPTH_SCALE="$2"
      shift 2
      ;;
    --depth_l1_weight_init)
      DEPTH_L1_WEIGHT_INIT="$2"
      shift 2
      ;;
    --depth_l1_weight_final)
      DEPTH_L1_WEIGHT_FINAL="$2"
      shift 2
      ;;
    --no_eval)
      EVAL=0
      shift
      ;;
    --no_save_training_vis)
      SAVE_TRAINING_VIS=0
      shift
      ;;
    --save_training_vis_iteration)
      SAVE_TRAINING_VIS_ITERATION="$2"
      shift 2
      ;;
    --skip_init)
      SKIP_INIT=1
      shift
      ;;
    --overwrite|--force)
      OVERWRITE=1
      shift
      ;;
    --start_round|--resume_from_round)
      START_ROUND="$2"
      shift 2
      ;;
    --removal_config_template)
      REMOVAL_CONFIG_TEMPLATE="$2"
      shift 2
      ;;
    --inpaint_config_template)
      INPAINT_CONFIG_TEMPLATE="$2"
      shift 2
      ;;
    --top_k_ref_views)
      TOP_K_REF_VIEWS="$2"
      shift 2
      ;;
    --simple_lama_device)
      SIMPLE_LAMA_DEVICE="$2"
      shift 2
      ;;
    --mask_threshold)
      MASK_THRESHOLD="$2"
      shift 2
      ;;
    --mask_dilation)
      MASK_DILATION="$2"
      shift 2
      ;;
    --intersect_top_m)
      INTERSECT_TOP_M="$2"
      shift 2
      ;;
    --intersect_cache_size)
      INTERSECT_CACHE_SIZE="$2"
      shift 2
      ;;
    --background_id)
      BACKGROUND_ID="$2"
      shift 2
      ;;
    --finetune_iteration)
      FINETUNE_ITERATION="$2"
      shift 2
      ;;
    --storage_mode)
      STORAGE_MODE="$2"
      shift 2
      ;;
    --render_intermediate)
      RENDER_INTERMEDIATE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "Unknown argument: $1"
      ;;
  esac
done

[[ -n "$SOURCE_PATH" ]] || fail "missing -s/--source_path"
[[ -n "$MODEL_PATH" ]] || fail "missing -m/--model_path"
[[ -d "$SOURCE_PATH" ]] || fail "source path does not exist: $SOURCE_PATH"
[[ -d "$SOURCE_PATH/object_mask" ]] || fail "dataset must contain object_mask/: $SOURCE_PATH/object_mask"
[[ -f "$TRAIN_CONFIG" ]] || fail "train config not found: $TRAIN_CONFIG"
[[ -z "$WORKFLOW_CONFIG" || -f "$WORKFLOW_CONFIG" ]] || fail "workflow config not found: $WORKFLOW_CONFIG"
[[ -z "$REMOVAL_CONFIG_TEMPLATE" || -f "$REMOVAL_CONFIG_TEMPLATE" ]] || fail "removal config template not found: $REMOVAL_CONFIG_TEMPLATE"
[[ -z "$INPAINT_CONFIG_TEMPLATE" || -f "$INPAINT_CONFIG_TEMPLATE" ]] || fail "inpaint config template not found: $INPAINT_CONFIG_TEMPLATE"
[[ "$START_ROUND" =~ ^[0-9]+$ ]] || fail "--start_round must be a non-negative integer: $START_ROUND"
case "$STORAGE_MODE" in
  full|lite|minimal)
    ;;
  *)
    fail "--storage_mode must be one of: full, lite, minimal. Got: $STORAGE_MODE"
    ;;
esac

if (( SKIP_INIT == 0 )) && [[ -z "$WORKFLOW_CONFIG" ]] && ((${#TARGET_GROUPS[@]} == 0)) && ((${#TARGET_IDS[@]} == 0)); then
  fail "--workflow_config/--config_file, --target_groups, or --target_ids is required unless --skip_init is used"
fi

for target_id in "${TARGET_IDS[@]}"; do
  [[ "$target_id" =~ ^[0-9]+$ ]] || fail "target id must be a non-negative integer: $target_id"
done

PIPELINE_START_TIME="$(date +%s)"
PIPELINE_START_TEXT="$(date '+%F %T')"

ensure_model_ready() {
  [[ -d "$MODEL_PATH/point_cloud" ]] || fail "trained point_cloud directory not found: $MODEL_PATH/point_cloud"

  shopt -s nullglob
  local iter_dirs=("$MODEL_PATH"/point_cloud/iteration_*)
  shopt -u nullglob
  ((${#iter_dirs[@]} > 0)) || fail "no point_cloud/iteration_* directory found in $MODEL_PATH"

  local has_classifier=0
  local dir
  for dir in "${iter_dirs[@]}"; do
    if [[ -f "$dir/classifier.pth" ]]; then
      has_classifier=1
      break
    fi
  done
  ((has_classifier == 1)) || fail "no classifier.pth found under $MODEL_PATH/point_cloud/iteration_*"
}

run_train() {
  if [[ -d "$MODEL_PATH/point_cloud" && "$ALLOW_EXISTING_TRAIN_OUTPUT" -eq 0 ]]; then
    fail "$MODEL_PATH already has point_cloud/. Use --skip_train or --allow_existing_train_output."
  fi

  local train_args=(
    "${PYTHON_CMD[@]}" "$SCRIPT_DIR/train.py"
    -s "$SOURCE_PATH"
    -m "$MODEL_PATH"
    --config_file "$TRAIN_CONFIG"
    --iterations "$ITERATIONS"
    --save_interval "$SAVE_INTERVAL"
    --checkpoint_interval "$CHECKPOINT_INTERVAL"
    --test_interval "$TEST_INTERVAL"
    --densification_interval "$DENSIFICATION_INTERVAL"
    --lambda_normal_render_depth "$LAMBDA_NORMAL_RENDER_DEPTH"
    --lambda_mask_entropy "$LAMBDA_MASK_ENTROPY"
  )

  if (( EVAL )); then
    train_args+=(--eval)
  fi
  if (( USE_DEPTH_LOSS )); then
    if [[ -z "$DEPTHS" ]]; then
      DEPTHS="$SOURCE_PATH/depth"
    fi
    train_args+=(
      --use_depth_loss
      --depths "$DEPTHS"
      --depth_scale "$DEPTH_SCALE"
      --depth_l1_weight_init "$DEPTH_L1_WEIGHT_INIT"
      --depth_l1_weight_final "$DEPTH_L1_WEIGHT_FINAL"
    )
  fi
  if (( SAVE_TRAINING_VIS )); then
    train_args+=(--save_training_vis --save_training_vis_iteration "$SAVE_TRAINING_VIS_ITERATION")
  fi

  echo "[$(date '+%F %T')] stage=train"
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" "${train_args[@]}"
}

run_init() {
  local init_args=(
    "${PYTHON_CMD[@]}" "$SCRIPT_DIR/iterative_inpaint_3dgic.py" init
    -s "$SOURCE_PATH"
    -m "$MODEL_PATH"
    --base_model_path "$MODEL_PATH"
    --type render
  )

  if [[ -n "$WORKFLOW_CONFIG" ]]; then
    init_args+=(--workflow_config "$WORKFLOW_CONFIG")
  fi
  if ((${#TARGET_GROUPS[@]} > 0)); then
    init_args+=(--target_groups "${TARGET_GROUPS[@]}")
  elif ((${#TARGET_IDS[@]} > 0)); then
    init_args+=(--target_ids "${TARGET_IDS[@]}")
  fi
  if [[ -n "$REMOVAL_CONFIG_TEMPLATE" ]]; then
    init_args+=(--removal_config_template "$REMOVAL_CONFIG_TEMPLATE")
  fi
  if [[ -n "$INPAINT_CONFIG_TEMPLATE" ]]; then
    init_args+=(--inpaint_config_template "$INPAINT_CONFIG_TEMPLATE")
  fi
  if [[ -n "$TOP_K_REF_VIEWS" ]]; then
    init_args+=(--top_k_ref_views "$TOP_K_REF_VIEWS")
  fi
  if [[ -n "$SIMPLE_LAMA_DEVICE" ]]; then
    init_args+=(--simple_lama_device "$SIMPLE_LAMA_DEVICE")
  fi
  if [[ -n "$MASK_THRESHOLD" ]]; then
    init_args+=(--mask_threshold "$MASK_THRESHOLD")
  fi
  if [[ -n "$MASK_DILATION" ]]; then
    init_args+=(--mask_dilation "$MASK_DILATION")
  fi
  if [[ -n "$INTERSECT_TOP_M" ]]; then
    init_args+=(--intersect_top_m "$INTERSECT_TOP_M")
  fi
  if [[ -n "$INTERSECT_CACHE_SIZE" ]]; then
    init_args+=(--intersect_cache_size "$INTERSECT_CACHE_SIZE")
  fi
  if [[ -n "$BACKGROUND_ID" ]]; then
    init_args+=(--background_id "$BACKGROUND_ID")
  fi
  if [[ -n "$FINETUNE_ITERATION" ]]; then
    init_args+=(--finetune_iteration "$FINETUNE_ITERATION")
  fi
  if (( OVERWRITE )); then
    init_args+=(--overwrite)
  fi

  echo "[$(date '+%F %T')] stage=init-workflow"
  "${init_args[@]}"
}

workflow_round_count() {
  "${PYTHON_CMD[@]}" -c 'import json, pathlib, sys; p = pathlib.Path(sys.argv[1]) / "iterative_3dgic" / "workflow.json"; w = json.load(open(p)); print(len(w.get("rounds", w.get("target_ids", []))))' "$MODEL_PATH"
}

run_rounds() {
  local round_count
  [[ -f "$MODEL_PATH/iterative_3dgic/workflow.json" ]] || fail "workflow not found: $MODEL_PATH/iterative_3dgic/workflow.json"
  round_count="$(workflow_round_count)"

  if (( START_ROUND >= round_count )); then
    fail "--start_round must be smaller than round count: start_round=$START_ROUND round_count=$round_count"
  fi

  local round_index
  for (( round_index=START_ROUND; round_index<round_count; round_index++ )); do
    local round_args=(
      "${PYTHON_CMD[@]}" "$SCRIPT_DIR/iterative_inpaint_3dgic.py" run-round
      -m "$MODEL_PATH"
      --round_index "$round_index"
      --storage_mode "$STORAGE_MODE"
    )
    if (( OVERWRITE )); then
      round_args+=(--overwrite)
    fi
    if (( RENDER_INTERMEDIATE )); then
      round_args+=(--render_intermediate)
    fi

    echo "[$(date '+%F %T')] stage=run-round round_index=${round_index}"
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" "${round_args[@]}"
  done
}

run_status() {
  echo "[$(date '+%F %T')] stage=status"
  "${PYTHON_CMD[@]}" "$SCRIPT_DIR/iterative_inpaint_3dgic.py" status -m "$MODEL_PATH"
}

if (( SKIP_TRAIN )); then
  echo "[$(date '+%F %T')] stage=train skipped"
else
  run_train
fi

ensure_model_ready

if (( SKIP_INIT )); then
  echo "[$(date '+%F %T')] stage=init-workflow skipped"
else
  run_init
fi

run_rounds
run_status
write_time_consuming


# 用法
# bash pipeline.sh \                                                                                                                 ─╯
#   -s ../../siga26/data/figurines \ 
#   -m ../../siga26/output/figurines/3dgic \
#   --workflow_config configs/iterative_inpaint/workflow_figurines.json \ 
#   --storage_mode minimal \
#   --overwrite
