import os,json,tqdm
import torch,joblib
import argparse
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
import warnings
from dataclasses import dataclass, field
warnings.filterwarnings('ignore')
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from safetensors import safe_open
from PIL import Image
torch.set_printoptions(threshold=5000)
import requests
from PIL import Image
from io import BytesIO
import re
import my_transformers
from typing import Dict, Optional, Sequence, List
import deepspeed

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

#os.environ['CUDA_VISIBLE_DEVICES']='1'

@dataclass
class ModelArguments:
    train_fusion: bool = field(default=False)

@dataclass
class DataArguments:
    image_aspect_ratio: str = 'square'


@dataclass
class TrainingArguments(my_transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    save_json: str = ""
    model_path: str = ""
    vision_tower: str = ""
    read_json: str = ""
    harm_image_files: str = ""
    output_path: str = ""

if __name__ == '__main__':
    parser = my_transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    #device = training_args.device
    device = torch.device("cuda:4")

    model_path = training_args.model_path
    data_list = []
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path),
        vision_tower=training_args.vision_tower,
        device=device,
        type_in=torch.float16,
    )

    with open(training_args.read_json, 'r') as f1:
        for line in f1:
            data_line = json.loads(line)

            text = data_line['response']
            prompt = '''Describe this image.'''
            # prompt = '''Describe this image and continue this content. Reference text: {}'''.format(text)
            qs = prompt
            image_file = data_line['image']
            args = type('Args', (), {
                "model_path": model_path,
                "model_base": None,
                "model_name": get_model_name_from_path(model_path),
                "query": prompt,
                "conv_mode": None,
                "image_file": image_file,
                "sep": ",",
                "temperature": 0,
                "top_p": None,
                "num_beams": 1,
                "max_new_tokens": 128
            })()
            model_name = get_model_name_from_path(args.model_path)
            image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            if IMAGE_PLACEHOLDER in qs:
                if model.config.mm_use_im_start_end:
                    qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
                else:
                    qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
            else:
                if model.config.mm_use_im_start_end:
                    qs = image_token_se + "\n" + qs
                    #qs = image_token_se + "\n" + 'Retrieved Text: '+ text +'. '+ qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
                    #qs = DEFAULT_IMAGE_TOKEN + "\n" + 'Retrieved Text: '+ text +'. '+ qs

            if "llama-2" in model_name.lower():
                conv_mode = "llava_llama_2"
            elif "mistral" in model_name.lower():
                conv_mode = "mistral_instruct"
            elif "v1.6-34b" in model_name.lower():
                conv_mode = "chatml_direct"
            elif "v1" in model_name.lower():
                conv_mode = "llava_v1"
            elif "mpt" in model_name.lower():
                conv_mode = "mpt"
            else:
                conv_mode = "llava_v0"

            if args.conv_mode is not None and conv_mode != args.conv_mode:
                print(
                    "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                        conv_mode, args.conv_mode, args.conv_mode
                    )
                )
            else:
                args.conv_mode = conv_mode

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            image_files = image_parser(args)
            images = load_images(image_files)
            image_sizes = [x.size for x in images]
            images_tensor = process_images(
                images,
                image_processor,
                model.config
            ).to(device, dtype=torch.float16)

            input_ids = (
                tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0).to(device)
            )

            images_tensor_zero = torch.zeros(images_tensor.shape).to(images_tensor.device, dtype=torch.float16)

            text_modality_ids = tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            ).input_ids.to(input_ids.device)

            torch.cuda.empty_cache()
            with torch.inference_mode():
                logits_dict, att_dict, hidden_dict = model(
                    input_ids=input_ids,
                    output_attentions=True,
                    output_hidden_states=True,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    return_dict=False,
                    return_analysis=True,
                    with_image=True,
                    analysis_inference=True,
                )
                outputs = ''

            dic_temp = {'image': data_line['image'], 'response': outputs,'promt':prompt,
                        'logits_dict':logits_dict,'att_dict':att_dict,'hidden_dict': hidden_dict,'image_token_pos':input_ids.shape[-1]}
            data_list.append(dic_temp)
            if len(data_list) % 100 == 0:
                joblib.dump(data_list, training_args.output_path+'analysis_data_llava_v1-6_mistral_vlm')

