import argparse

import numpy as np
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder_cls_ctr_cluster import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
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
instruction_pool=[
"Comparing the left part and right part of this image, which part is more suitable for the product '{}' ?",
"The left and right part of this image is one advertising image for the product '{}', respectively, which is preferred by the user?",
"Which part will bring more click-through rate in this image for product '{}' ?",
"Between the left and right sections of this image, which one is more appropriate for showcasing the product '{}'?",
"Considering the left and right halves of this image, which side better represents the product '{}'?",
"Which side of this image, left or right, is more effective for advertising the product '{}'?",
"For the product '{}', which part of the image, left or right, is more appealing?",
"When looking at the left and right portions of this image, which part is more suitable for promoting the product '{}'?",
"In this image, which side, left or right, is preferred by users for the product '{}'?",
"Which half of this image, left or right, is more likely to attract user preference for the product '{}'?",
"Which section of this image, left or right, is expected to generate a higher click-through rate for the product '{}'?",
"For the product '{}', which side of the image do users find more engaging, left or right?",
"Between the left and right sides of this image, which one is anticipated to drive more clicks for the product '{}'?",
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
    qs =caption


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
    print(prompt)
    return prompt

def get_substring_between_words(s, start_word, end_word):
    start_index = s.find(start_word)
    end_index = s.find(end_word, start_index + len(start_word))

    if start_index == -1 or end_index == -1 or start_index >= end_index:
        return None

    return s[start_index + len(start_word) : end_index]

def get_data(input_data_path, log_file_path):
    
    sku_msg_merge = {}
    with open(log_file_path, 'r') as f:
        for x in f:
            trans_save_path, index, sd_image_path, _, prompt_txt = x.strip().split('\t')
            key = '{}_{}'.format(index, prompt_txt)
            if sku_msg_merge.get(key) is None:
                sku_msg_merge[key] = sd_image_path

    with open(input_data_path, 'r') as f:
        cluster_list = json.load(f)

    new_cluster_list = []
    for idx, item in enumerate(cluster_list):

        index = item['id']
        sku = item['sku_id']

        if isinstance(item["prompt_answer"], list):
            current_prompt = item["prompt_answer"]
            current_prompt = [
                per_prompt.strip()
                .replace('"', "")
                .replace("\n", "")
                .replace(".", "")
                for per_prompt in current_prompt
            ]
            if len(current_prompt) == len(item["prompt_answer"]):
                sd_answer_list = []
                for prompt_txt in current_prompt:
                    key = '{}_{}'.format(index, prompt_txt)
                    if sku_msg_merge.get(key) is not None:
                        sd_answer_list.append(sku_msg_merge[key])
                    else:
                        print(key)
                if len(sd_answer_list) == len(item["prompt_answer"]):
                    item["sd_answer"] = sd_answer_list
                    item["prompt_answer"] = current_prompt
                    new_cluster_list.append(item)
        else:
            prompt_txt = (
                item["prompt_answer"]
                .strip()
                .replace('"', "")
                .replace("\n", "")
                .replace(".", "")
            )
            key = '{}_{}'.format(index, prompt_txt)
            if sku_msg_merge.get(key) is not None:
                item["sd_answer"] = sku_msg_merge[key]
                item["prompt_answer"] = prompt_txt
                new_cluster_list.append(item)

    out_json_path = input_data_path.replace('sd_prompt.json', 'reward.json')
    with open(out_json_path, 'w') as f:
        json.dump(new_cluster_list, f, ensure_ascii=False, indent=4)

    return new_cluster_list



def generate_prompt(args):
    disable_torch_init()
    process_group_kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=540000))
    accelerator = Accelerator(kwargs_handlers=[process_group_kwargs])

    if  args.model_path == "None":
        args.model_path = args.model_base
        args.model_base = None
        print("change args.model_path to args.model_base")
    model_name = get_model_name_from_path(args.model_path)

    task_list = get_data(args.base_data_path, args.log_file_path)
    print("total num: \n", len(task_list))
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, device=accelerator.process_index
    )
    mm_use_im_start_end = model.config.mm_use_im_start_end

    accelerator.wait_for_everyone()
    start=time.time()
    with accelerator.split_between_processes(task_list) as prompts:
        results = []
        for num, item in enumerate(prompts):
            if len(item["sd_answer"]) < 2 or len(item["prompt_answer"]) < 2:
                continue

            prompt = process_prompt(item["reward_question"], args, mm_use_im_start_end, model_name)
            input_ids = (
                tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .cuda()
            )
            item['reward_question_prompt'] = prompt
            if item.get('image') is None:
                item['image'] = item["trans_save_path"]

            if item.get("group_feature") is not None:
                user_infos = np.array([item['group_feature']])
            else:
                user_infos = None
            if item.get("sku_ctr") is not None:
                sku_ctrs = [item['sku_ctr']]
            else:
                sku_ctrs = None

            image_1 = Image.open(item["sd_answer"][0].replace('/media/cfs/', '/home/')).convert("RGB")
            image_2 = Image.open(item["sd_answer"][1].replace('/media/cfs/', '/home/')).convert("RGB")

            image_1 = image_1.resize((512, 512))
            image_2 = image_2.resize((512, 512))

            image_1_np = np.array(image_1)
            image_2_np = np.array(image_2)

            images = [Image.fromarray(np.concatenate([image_1_np, image_2_np], axis=1))]
            image_sizes = [x.size for x in images]

            images_tensor = process_images(
                images,
                image_processor,
                model.config
            ).to(model.device, dtype=torch.bfloat16)


            print('prompt', prompt)

            with torch.inference_mode():
                model_output = model.forward(
                    input_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    return_dict=True,
                    user_infos=user_infos,
                    sku_ctrs=sku_ctrs
                )
            print(model_output.logits)


            if model_output.logits[0,0]>=model_output.logits[0,1]:
                chosen_conversations = [
                    {
                        "from": "human",
                        "value": item['prompt_question'],
                        "reward_score":model_output.logits[0, 0].tolist(),
                        "target": "left"
                    },
                    {
                        "from": "gpt",
                        "value": item["prompt_answer"][0]
                    }
                ]

                reject_conversations = [
                    {
                        "from": "human",
                        "value": item['prompt_question'],
                        "reward_score": model_output.logits[0, 1].tolist(),
                        "target": "right"

                    },
                    {
                        "from": "gpt",
                        "value": item["prompt_answer"][1]
                    }
                ]
            else:
                chosen_conversations = [
                    {
                        "from": "human",
                        "value": item['prompt_question'],
                        "reward_score": model_output.logits[0, 1].tolist(),
                        "target": "right"
                    },
                    {
                        "from": "gpt",
                        "value": item["prompt_answer"][1]
                    }
                ]

                reject_conversations = [
                    {
                        "from": "human",
                        "value": item['prompt_question'],
                        "reward_score": model_output.logits[0, 0].tolist(),
                        "target": "left"
                    },
                    {
                        "from": "gpt",
                        "value": item["prompt_answer"][0]
                    }
                ]
            item['chosen_conversations'] = chosen_conversations
            item['reject_conversations'] = reject_conversations
            results.append(item)

    results_gathered=gather_object(results)
    formatted_nested_data = json.dumps(results_gathered, indent=4, ensure_ascii=False)
    if accelerator.is_main_process:
        timediff=time.time()-start
        with open(args.output_data_path, 'w') as file:
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
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument(
        "--log_file_path",
        type=str,
    )
    parser.add_argument("--base_data_path", type=str)
    parser.add_argument("--output_data_path", type=str)
    args = parser.parse_args()
    print(args)
    generate_prompt(args)
    