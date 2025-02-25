#!/bin/bash
WORK_DIR=`pwd` # == ${ALGO_DIR}
CODE_DIR=${WORK_DIR}
python ${CODE_DIR}/llava/tox_analysis/read_analysis_llava.py \
    --image_path ${harm_data}/output/analysis_data_llava_v1-6_mistral_vlm \
    --no_image_path ${harm_data}/output/analysis_data_llava_v1-6_mistral_vlm_no_image