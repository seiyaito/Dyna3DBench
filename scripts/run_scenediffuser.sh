#!/usr/bin/env bash

set -e

SCRIPT_DIR=$(cd $(dirname $0); pwd)

SCENE_LIST=${SCRIPT_DIR}/ScanNet/Tasks/Benchmark/scannetv2_val.txt
OUTPUT_PATH=${DYNA3DBENCH}/motions
SEED=2024

while [[ $# -gt 0 ]]; do
  case $1 in
    -l|--list)
      SCENE_LIST="$2"
      shift 2
      ;;
    -s|--seed)
      SEED=$2
      shift 2
      ;;
    -o|--output)
      OUTPUT_PATH="$2"
      shift 2
      ;;
    *)
      echo "Usage: $0 [-l|--list SCENE_LIST] [-s|--seed SEED] [-o|--output OUTPUT_PATH]"
      exit 1
      ;;
  esac
done

SCENE_DIFFUSER_DIR=/Scene-Diffuser
CONFIG_TEMPLATE=${SCRIPT_DIR}/motion_gen.yaml
CONFIG_FILE=${SCENE_DIFFUSER_DIR}/configs/task/motion_gen.yaml
BODY_SEGMENTS_DIR=/data/PROX/body_segments
CHECKPOINT_DIR=/data/checkpoints/2022-11-09_12-54-50_MotionGen_ddm_T200_lr1e-4_ep300

. ${SCRIPT_DIR}/download_scannet_tool.sh

OUTPUT_FOLDER=${OUTPUT_PATH}/PROX
mkdir -p ${OUTPUT_FOLDER}

mkdir -p ${OUTPUT_PATH}
mkdir -p ${CHECKPOINT_DIR}/eval
if [ ! -e "${CHECKPOINT_DIR}/eval/final" ]; then
  ln -s ${OUTPUT_PATH} ${CHECKPOINT_DIR}/eval/final
fi

# Cam2World
python ${SCRIPT_DIR}/../convert_cam2world.py \
  --dataset_root ${SCANNETV2} \
  --list ${SCENE_LIST} \
  --output_dir ${OUTPUT_FOLDER}/cam2world \
  --seed ${SEED}

# body segments
if [ ! -e "${OUTPUT_FOLDER}/body_segments" ]; then
  ln -s ${BODY_SEGMENTS_DIR} ${OUTPUT_FOLDER}/body_segments
fi

# Scenes
mkdir  -p ${OUTPUT_FOLDER}/scenes
while IFS= read -r scene; do
  if [ -n "$scene" ]; then
    if [ ! -f "${OUTPUT_FOLDER}/scenes/${scene}.ply" ]; then
      ln -s ${SCANNETV2}/scans/$scene/${scene}_vh_clean_2.ply ${OUTPUT_FOLDER}/scenes/${scene}.ply
    fi
  fi
done < "${SCENE_LIST}"

# Preprocess scene
while IFS= read -r scene; do
  if [ -f "${OUTPUT_FOLDER}/preprocess_scenes/${scene}.npy" ]; then
    continue
  fi

  python ${SCRIPT_DIR}/../preprocess_scene.py \
    --scene_dir ${OUTPUT_FOLDER}/scenes \
    --preprocess_scenes_dir ${OUTPUT_FOLDER}/preprocess_scenes \
    --scene_id ${scene}
done < "${SCENE_LIST}"


# SDF
mkdir -p ${OUTPUT_FOLDER}/sdf
while IFS= read -r scene; do
  if [ -n "$scene" ]; then
    INPUT_FOLDER="${SCANNETV2}/scans/$scene"

    if [ -f "${OUTPUT_FOLDER}/sdf/${scene}.json" ]; then
        continue
    fi

    python ${SCRIPT_DIR}/../convert_mesh_to_sdf.py \
      --input_mesh ${OUTPUT_FOLDER}/scenes/${scene}.ply \
      --output_dir ${OUTPUT_FOLDER}/sdf 
  fi
done < "${SCENE_LIST}"

if [ -f "${CONFIG_FILE}" ]; then
  mv ${CONFIG_FILE} ${SCENE_DIFFUSER_DIR}/configs/task/motion_gen.default.yaml
fi

while IFS= read -r scene; do
  if [ -n "$scene" ]; then
    if [ -d ${OUTPUT_PATH}/${scene} ]; then
      continue
    fi
    # Config
    if [ -f ${CONFIG_FILE} ]; then
      rm ${CONFIG_FILE}
    fi

    while IFS= read -r line; do
      if expanded_line=$(eval "echo \"$line\"" 2>/dev/null); then
        echo "$expanded_line" >> "$CONFIG_FILE"
      else
        echo "$line" >> "$CONFIG_FILE"
      fi
    done < "${CONFIG_TEMPLATE}"

    # run Scene-Diffuser
    cd ${SCENE_DIFFUSER_DIR}
    bash scripts/motion_gen/sample.sh ${CHECKPOINT_DIR}

  fi
done < "${SCENE_LIST}"

while IFS= read -r scene; do
  if [ -n "$scene" ]; then
    scene_path="${OUTPUT_PATH}/${scene}"
    if [ -d "$scene_path" ]; then
      continue
    fi

    for motion in $(ls "$OUTPUT_PATH" | grep -v PROX); do
      motion_path="${OUTPUT_PATH}/${motion}"
      for res in $(ls "$motion_path" | grep -v log | grep "$scene"); do
        mv "${motion_path}/${res}" "$scene_path"
        mv "${motion_path}/sample.log" "$scene_path"
        rm -rf "$motion_path"
      done
    done
  fi
done < "${SCENE_LIST}"
