#!/usr/bin/env bash

set -e

SCRIPT_DIR=$(cd $(dirname $0); pwd)

SCENE_LIST=${SCRIPT_DIR}/ScanNet/Tasks/Benchmark/scannetv2_val.txt
OUTPUT_PATH=${DYNA3DBENCH}/images
MOTION_PATH=${DYNA3DBENCH}/motions
SENS_PATH=${DYNA3DBENCH}/sens
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

. ${SCRIPT_DIR}/download_scannet_tool.sh

xvfb-run python synthesize.py \
  --scannet_root ${SCANNETV2}/scans \
  --scene_list ${SCENE_LIST} \
  --motion_mesh_dir ${MOTION_PATH} \
  --output_dir ${OUTPUT_PATH} \
  --save_mask \
  --num_samples 0 1 2 \
  --seed ${SEED}


while IFS= read -r scene; do
  if [ -n "$scene" ]; then
    INPUT_FOLDER="${SCANNETV2}/scans/$scene"

    if [ -f "${SENS_PATH}/${scene}/${scene}.json" ]; then
        continue
    fi

    for n in ${OUTPUT_PATH}/${scene}/*; do
      mkdir -p ${SENS_PATH}/$(basename "$n")/${scene}
      python ${SCRIPT_DIR}/../sensor_data.py \
        ${SCANNETV2}/scans/${scene}/${scene}.sens \
        --rendered_images ${n} \
        --output ${SENS_PATH}/$(basename "$n")/${scene}/${scene}.sens
    done
  fi
done < "${SCENE_LIST}"
