import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import numpy as np
from tqdm import tqdm


import argparse
import torch
if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder_user import load_pretrained_model

from llava.utils import disable_torch_init
from llava.mm_utils_user import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image
import time
import requests
from PIL import Image
from io import BytesIO
import re
import json
import random
from accelerate import Accelerator
from accelerate.utils import gather_object
from accelerate import InitProcessGroupKwargs
import datetime
import numpy as np

import pdb

instruction_pool = [
    "Design a concise Stable Diffusion prompt that takes the product caption '{}' and product image as inspiration to generate an appealing advertising image background for this product. (Directly give me the prompt without prefix explanation).",
    "Produce a short diffusion prompt considering the critical information in product caption '{}' and product image to create an advertisement background. (Directly give me the prompt without prefix explanation).",
    "Construct a brief background generation prompt to produce an advertising image based on product caption '{}' and product image.  (Directly give me the prompt without prefix explanation).",
    "Generate a short Stable Diffusion prompt that leverage the product caption '{}' and the visual elements of the product image to output an advertising background that underscores the product's attractiveness. (Directly give me the prompt without prefix explanation).",
    "Provide a succinct text to background diffusion model prompt suitable for this product according to caption '{}' and product image. (Directly give me the prompt without prefix explanation).",
    "Develop a compact prompt for Stable Diffusion to craft a background for an ad, using the product caption '{}' and the accompanying product image as creative influences. (Directly give me the prompt without prefix explanation).",
    "Draft a distilled text to image prompt to fabricate an ad background, infusing the essence of '{}' from the product caption and the picture to accentuate the product's features. (Directly give me the prompt without prefix explanation).",
    "Based on this product image and product caption '{}', formulate a brief diffusion prompt to synthesize a background tailored for advertising purposes. (Directly give me the prompt without prefix explanation).",
    "Take advantage of this product with its title '{}', build a succinct prompt for diffusion model aimed at generating an advertisement background. (Directly give me the prompt without prefix explanation).",
    "Make use of product title '{}' along with its image, assemble a terse stable diffusion model prompt to render an ad background that complements and highlights the product. (Directly give me the prompt without prefix explanation).",
]
add_human_prompt_pool = [
    ' Your answer should be targeted at consumer <user>.',
    ' Please consider the characteristics of consumer <user> and answer.',
    ' Consumers with characteristic <user> is your target group.',
    ' Please pay attention to the consumer with characteristic <user> and make the most suitable choice for him.'


]



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


def eval_model(args):
    import deepspeed
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )
    
    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

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
    ).to(model.device, dtype=torch.bfloat16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    print(outputs)

def batch_list(samples, batch_size):
            """Batch the given list into sublists of specified max size."""
            return [samples[i:i + batch_size] for i in range(0, len(samples), batch_size)]

def process_prompt(caption, args, mm_use_im_start_end, model_name):


    instruct = random.choice(instruction_pool)  
    qs = instruct.format(caption.strip().replace('"', ""))
    user_instruct = random.choice(add_human_prompt_pool)
    qs += user_instruct

    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

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
    prompt = conv.get_prompt().replace("A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: ", "")

    return prompt


def generate_prompt(args, thr=0.8):


    disable_torch_init()
    
    process_group_kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=540000))
    accelerator = Accelerator(kwargs_handlers=[process_group_kwargs])

    if  args.model_path == "None":
        args.model_path = args.model_base
        args.model_base = None
        print("change args.model_path to args.model_base")
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, device=accelerator.process_index
    )
    mm_use_im_start_end = model.config.mm_use_im_start_end
    
    f = open(args.base_data_path, 'r')
    content = f.read()
    a = json.loads(content)
    print("total num: ", len(a))
    if args.batch_samples:
        a = random.sample(a, args.batch_samples)
        print("random select samples: ", args.batch_samples)

    accelerator.wait_for_everyone()
    start=time.time()


    with accelerator.split_between_processes(a) as prompts:
        new_list = []

        total = len(prompts)

        for num, item in tqdm(enumerate(prompts), total=total, desc="Generating prompts",disable=not accelerator.is_local_main_process):


            new_sample=item

            prompt = process_prompt(item["new_caption"], args, mm_use_im_start_end, model_name)

            input_ids = (tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda())
            new_sample['prompt_question'] = prompt


            if 'trans_save_path' in item:

                image_file = item['trans_save_path']
                if item.get('image') is None:
                    item['image'] = item["trans_save_path"]

                images = [Image.open(image_file).convert("RGB") ]
                image_sizes = [x.size for x in images]

                images_tensor = process_images(
                    images,
                    image_processor,
                    model.config
                ).to(model.device, dtype=torch.float16)
            else:

                crop_size = model.get_vision_tower().image_processor.crop_size
                images_tensor = torch.zeros(3, crop_size['height'], crop_size['width']).unsqueeze(0).to(model.device, dtype=torch.float16)
                image_sizes = [(crop_size['height'], crop_size['width']) for _ in range(1)]

            if item.get("group_feature") is not None:
                users = torch.from_numpy(np.array(item['group_feature'])).unsqueeze(0).to(model.device, dtype=torch.float16)

            else:
                users = torch.zeros(1, 16).unsqueeze(0).to(model.device, dtype=torch.float16)




            if args.generate_nums>1:
                outputs = []

            
            args.max_new_tokens = 512
            for i in range(args.generate_nums):
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=images_tensor,
                        image_sizes=image_sizes,
                        do_sample=(True if args.temperature > 0 else False),
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True,
                        users=users
                    )

                output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                if args.generate_nums > 1:
                    outputs.append(output)
                else:
                    outputs = output

                print('generate:', output)

                
            new_sample["prompt_answer"] = outputs
            new_list.append(new_sample)
            print("index: ", num, "input_and_out: ", new_sample)

    results_gathered = gather_object(new_list)
    formatted_nested_data = json.dumps(results_gathered, indent=4, ensure_ascii=False)
    if accelerator.is_main_process:
        timediff = time.time() - start
        output_path = args.output_data_path
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        with open(args.output_data_path, "w", encoding="utf-8") as file:
            file.write(formatted_nested_data)
        print(f"time elapsed: {timediff}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str)
    parser.add_argument("--query", type=str)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_samples", type=int, default=None)
    parser.add_argument("--generate_nums", type=int, default=1)
    parser.add_argument("--base_data_path", type=str)
    parser.add_argument("--output_data_path", type=str)

    args = parser.parse_args()
    

    generate_prompt(args)
