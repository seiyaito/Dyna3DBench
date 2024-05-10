#!/usr/bin/env bash

set -e

SCRIPT_DIR=$(cd $(dirname $0); pwd)
SCANNET_TOOL_DIR=${SCRIPT_DIR}/ScanNet

if [ ! -d "${SCANNET_TOOL_DIR}" ]; then
    git clone https://github.com/ScanNet/ScanNet ${SCANNET_TOOL_DIR} \
      && cd ${SCANNET_TOOL_DIR} \
      && git checkout d0a4b9e09dc38b2570ca4016d2ad4f2d373b46a5 
fi
