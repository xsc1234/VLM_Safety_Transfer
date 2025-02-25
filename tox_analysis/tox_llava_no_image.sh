#!/bin/bash
WORK_DIR=`pwd` # == ${ALGO_DIR}
CODE_DIR=${WORK_DIR}

MODEL=${llava-v1.6-mistral}
VISION_TOWER=${clip-vit-large-patch14-336}
harm_data=${harm_data_path}
READ_JSON=${harm_data}/output/data_llava_v1-6_mistral_porn_vlm.json
pip install --user -r ${CODE_DIR}/req.txt
pip install --user --upgrade 'urllib3==1.26.7'
nvidia-smi
pip list

NUM_GPU=`python -c "import torch; print(torch.cuda.device_count())"`

python ${CODE_DIR}/llava/tox_analysis/analysis_llava_no_image.py \
    --model_path ${MODEL}  \
    --read_json ${READ_JSON} \
    --harm_image_files ${harm_data} \
    --vision_tower ${VISION_TOWER} \
    --output_path ${harm_data}/output/