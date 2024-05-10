#!/usr/bin/env bash

set -e

SCRIPT_DIR=$(cd $(dirname $0); pwd)

SCENE_LIST="${SCRIPT_DIR}/ScanNet/Tasks/Benchmark/scannetv2_val.txt"
OUTPUT_PATH="${DYNA3DBENCH}/reconstructions/0"
DATASET_PATH="${DYNA3DBENCH}/sens/0"
DATASET_NAME="static"

TEMP=$(getopt -o s:o:d:n: --long scene-list:,output-path:,dataset-path:,dataset-name: -n 'parse-options' -- "$@")
if [ $? != 0 ] ; then echo "Terminating..." >&2 ; exit 1 ; fi

eval set -- "$TEMP"

while true; do
  case "$1" in
    -s | --scene-list ) SCENE_LIST="$2"; shift 2 ;;
    -o | --output-path ) OUTPUT_PATH="$2"; shift 2 ;;
    -d | --dataset-path ) DATASET_PATH="$2"; shift 2 ;;
    -n | --dataset-name ) DATASET_NAME="$2"; shift 2 ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

SCFUSION_DIR=/SCFusion
MODEL_PATH=${SCFUSION_DIR}/Models/SI_ScanNet_0614.pt
LABEL_COLOR_PATH=${SCFUSION_DIR}/Files/LabelColorLists_SunCG11.txt

. ${SCRIPT_DIR}/download_scannet_tool.sh


if [ ! -f "${SCENE_LIST}" ]; then
  echo "Scene list file '$scene_list' not found"
  exit 1
fi

while IFS= read -r scene; do
  if [ -n "$scene" ]; then
    # Output Folder
    OUTPUT_FOLDER="${OUTPUT_PATH}/${DATASET_NAME}/${scene}"
    mkdir -p ${OUTPUT_FOLDER}
    
    # Config
    INPUT_FOLDER="${DATASET_PATH}/${scene}"
    CONFIG_FILE="${OUTPUT_FOLDER}/Config.txt"
    if [ -f "${CONFIG_FILE}" ]; then
      continue
    fi

    while IFS= read -r line; do
      eval "echo \"$line\"" >> "${CONFIG_FILE}"
    done < "${CONFIG_TEMPLATE}"

    # Reconstruct
     ${SCFUSION_DIR}/build/App/SCFusion/exe_scfusion_OFusionRGB1Label \
      ${CONFIG_FILE} \
      --useGTPose 2 \
      --useSC 1 \
      --pthOut ${OUTPUT_FOLDER}
    
    # Mesh
    mkdir -p ${OUTPUT_FOLDER}/mesh
    ${SCFUSION_DIR}/build/App/Map2Mesh/exe_Map2Mesh_OFusionRGB1Label \
      --pth_in ${OUTPUT_FOLDER}/ \
      --pth_out ${OUTPUT_FOLDER}/mesh \
      --labelColorPath ${LABEL_COLOR_PATH}
  fi
done < "${SCENE_LIST}"
