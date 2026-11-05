from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UNet2DConditionModel,
)
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import PIL
import cv2
import torch
import os
import dataclasses
from tqdm.auto import tqdm
from utils.utils import choose_scheduler, resize_and_canny, CannyDetector, concat_image
from utils.clip_text_custom_embedder import text_embeddings
from utils.attention_utils import (
    cross_attn_init,
    register_cross_attention_hook,
    attn_maps,
    get_net_attn_map,
    resize_net_attn_map,
    save_net_attn_map,
)
from random import choice
import random
from datetime import datetime
from .pipeline_StableDiffusionControlNetPipeline_extended import (
    StableDiffusionControlNetPipelineExtended,
)
import copy
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
import logging
from diffusers.optimization import get_scheduler
from torch.nn.parallel import DistributedDataParallel
import time
import json
import textwrap
from tqdm import tqdm
logger = get_logger(__name__, log_level="INFO")


@dataclasses.dataclass(frozen=False)
class TrainPolicyFuncData:
    tot_p_loss: float = 0
    tot_ratio: float = 0
    tot_kl: float = 0
    tot_grad_norm: float = 0


class T2I_CN:
    def __init__(self, args):
        # self.pipe=self.prepare_model(args)
        self.prompt = args.prompt
        self.negative_prompt = args.negative_prompt
        self.guidance_scale = args.guidance_scale
        self.num_inference_steps = args.num_inference_steps
        self.width = args.width
        self.height = args.height
        self.controlnet_conditioning_scale = args.controlnet_conditioning_scale
        self.clip_skip = args.clip_skip  # -1
        # self.clip_skip = self.clip_skip if self.clip_skip >= 1 else None
        self.batch_size = args.batch_size
        self.save_path = args.save_path
        # if not os.path.exists(self.save_path):
        #     os.mkdir(self.save_path)
        self.image_scale = args.image_scale
        self.preprocessor = CannyDetector(args.low_threshold, args.high_threshold)
        self.keep_loc = args.keep_loc
        self.save_concat = args.save_concat
        self.args = args
        self.log_file = args.log_file
        self.concat_path = os.path.realpath(self.save_path) + "-concat"
        if self.save_concat:
            os.makedirs(self.concat_path, exist_ok=True)
        #     if self.args.save_attn:
        #         self.concat_path=self.concat_path+'_att'
        #     if not os.path.exists(self.concat_path):
        #         os.mkdir(self.concat_path)

    def trans_pool(self):

        self.img_list = []
        self.prompt_list = []
        self.index_list = []

        f = open(self.args.data_path, "r")
        content = f.read()
        a = json.loads(content)
        sku_dict = {}
        for idx, item in enumerate(a):
            # current_prompt = item['answer'].strip().replace("\"", "").replace("\n", "")
            # current_image = item['image']
            # self.img_list.append(current_image)
            # self.prompt_list.append(current_prompt)
            index = item['id']
            sku = item['sku_id']
            sku_dict[sku] = ''
            if isinstance(item["prompt_answer"], list):
                current_prompt = item["prompt_answer"]
                current_prompt = [
                    per_prompt.strip()
                    .replace('"', "")
                    .replace("\n", "")
                    .replace(".", "")
                    for per_prompt in current_prompt
                ]
                self.prompt_list.extend(current_prompt)
                self.per_sku = len(current_prompt)
                current_image = item["trans_save_path"]
                repeated_image = [current_image for i in range(self.per_sku)]
                repeated_index = [index for i in range(self.per_sku)]
                self.img_list.extend(repeated_image)
                self.index_list.extend(repeated_index)
            else:
                current_prompt = (
                    item["prompt_answer"]
                    .strip()
                    .replace('"', "")
                    .replace("\n", "")
                    .replace(".", "")
                )
                current_image = item["trans_save_path"]
                self.img_list.append(current_image)
                self.prompt_list.append(current_prompt)
                self.index_list.append(index)
                self.per_sku = 1
            # print("current_prompt", current_prompt)
            # if "<|endoftext|>" in current_prompt[0]:
            #     print("<|endoftext|>!!!!!", current_prompt[0])
            # print(len(current_prompt[0]),len(current_prompt[1]),len(current_prompt[2]))
        print('sku count is', len(sku_dict))
        print('question count is', len(a))
        print('self.per_question count is', self.per_sku)
        print('trans_img_list count is', len(self.img_list))
        print('prompt_list count is', len(self.prompt_list))
        print('index_list count is', len(self.index_list))


    def prepare_model(self, args, device_map=None):
        if self.log_file is None:
            self.log_file = os.path.join(self.save_path, "log.txt")
        if self.args.save_attn:
            cross_attn_init()
        if os.path.isfile(args.controlnet_model_path):
            controlnet = ControlNetModel.from_single_file(
                args.controlnet_model_path,
                config_file=args.controlnet_config_path,
                device_map=device_map,
            )
        else:
            controlnet = ControlNetModel.from_pretrained(
                args.controlnet_model_path, use_safetensors=True, device_map=device_map
            )
        if os.path.isfile(args.base_model_path):
            pipe = StableDiffusionControlNetPipelineExtended.from_single_file(
                args.base_model_path,
                local_files_only=True,
                load_safety_checker=False,
                original_config_file=args.original_config_file,
                controlnet=controlnet,
                device_map=device_map,
            )
            # pipe = StableDiffusionControlNetPipeline.from_single_file(args.base_model_path, local_files_only=True,load_safety_checker=False, original_config_file=args.original_config_file,controlnet=controlnet)
        else:
            pipe = StableDiffusionControlNetPipelineExtended.from_pretrained(
                args.base_model_path,
                controlnet=controlnet,
                safety_checker=None,
                device_map=device_map,
            )
        if args.unet_path:
            load_model = UNet2DConditionModel.from_pretrained(args.unet_path)
            # model.register_to_config(**load_model.config)
            print("load unet weights!!!!")
            pipe.unet.load_state_dict(load_model.state_dict())
        if args.lora_model_path:
            pipe.load_lora_weights(args.lora_model_path)
            pipe.fuse_lora(lora_scale=args.lora_scale)
            print("load lora finished!!")
        pipe.scheduler = choose_scheduler(args.sampler_name).from_config(
            pipe.scheduler.config
        )
        # if self.args.save_attn:
        #     pipe.unet = register_cross_attention_hook(pipe.unet)
        pipe.set_progress_bar_config(disable=True)

        pipe.to("cuda")

        # if args.contrast:
        #     if os.path.isfile(args.new_controlnet_model_path):
        #         controlnet_new = ControlNetModel.from_single_file(args.new_controlnet_model_path,config_file=args.controlnet_config_path)
        #     else:
        #         controlnet_new = ControlNetModel.from_pretrained(args.new_controlnet_model_path, use_safetensors=True)
        #     if os.path.isfile(args.base_model_path):
        #         pipe_new = StableDiffusionControlNetPipelineExtended.from_single_file(args.base_model_path, local_files_only=True, load_safety_checker=False, original_config_file=args.original_config_file,controlnet=controlnet_new)
        #         # pipe = StableDiffusionControlNetPipeline.from_single_file(args.base_model_path, local_files_only=True,load_safety_checker=False, original_config_file=args.original_config_file,controlnet=controlnet)
        #     else:
        #         pipe_new = StableDiffusionControlNetPipelineExtended.from_pretrained(args.base_model_path, controlnet=controlnet_new, safety_checker=None)
        #     if args.lora_model_path:
        #         pipe_new.load_lora_weights(args.lora_model_path)
        #         pipe_new.fuse_lora(lora_scale=args.lora_scale)
        #         print("load lora finished!!")
        #     pipe_new.scheduler = choose_scheduler(args.sampler_name).from_config(pipe_new.scheduler.config)
        #     if self.args.save_attn:
        #         pipe_new.unet = register_cross_attention_hook(pipe_new.unet)
        #     pipe_new.to("cuda")

        #     if args.newnew_controlnet_model_path is not None:
        #         if os.path.isfile(args.newnew_controlnet_model_path):
        #             controlnet_newnew = ControlNetModel.from_single_file(args.newnew_controlnet_model_path, config_file=args.controlnet_config_path)
        #         else:
        #             controlnet_newnew = ControlNetModel.from_pretrained(args.newnew_controlnet_model_path, use_safetensors=True)
        #         if os.path.isfile(args.base_model_path):
        #             pipe_newnew = StableDiffusionControlNetPipelineExtended.from_single_file(args.base_model_path, local_files_only=True, load_safety_checker=False, original_config_file=args.original_config_file,controlnet=controlnet_new)
        #             # pipe = StableDiffusionControlNetPipeline.from_single_file(args.base_model_path, local_files_only=True,load_safety_checker=False, original_config_file=args.original_config_file,controlnet=controlnet)
        #         else:
        #             pipe_newnew = StableDiffusionControlNetPipelineExtended.from_pretrained(args.base_model_path, controlnet=controlnet_new, safety_checker=None)
        #         if args.lora_model_path:
        #             pipe_newnew.load_lora_weights(args.lora_model_path)
        #             pipe_newnew.fuse_lora(lora_scale=args.lora_scale)
        #             print("load lora finished!!")
        #         pipe_newnew.scheduler = choose_scheduler(args.sampler_name).from_config(pipe_newnew.scheduler.config)
        #         if self.args.save_attn:
        #             pipe_newnew.unet = register_cross_attention_hook(pipe_newnew.unet)
        #         pipe_newnew.to("cuda")
        #         return  pipe, pipe_new, pipe_newnew
        #     return pipe, pipe_new
        return pipe

    def concat_prompt(
        self,
        image,
        prompt,
        font_size=50,
        text_color=(0, 0, 0),
        bg_color=(255, 255, 255),
    ):
        image_width, image_height = image.size

        # Create a new image with space for the text (using an estimated height for the default font)
        text_height = 10 + font_size  # Estimate the height based on font size
        if image_height == 1024:
            extended_height = image_height + 360  # Add some padding below the text
            font_size = 50
        else:
            extended_height = image_height + 180
            font_size = 25
        new_image = Image.new("RGB", (image_width, extended_height), color=bg_color)
        new_image.paste(image, (0, 0))

        # Create a drawing context for the new image
        draw = ImageDraw.Draw(new_image)

        # Use the default font provided by PIL
        font = ImageFont.truetype(
            "path_to_Times_New_Roman.ttf",
            font_size,
        )
        # font = ImageFont.load_default()

        # Calculate text position (centered)
        # text_width, _ = draw.textsize(prompt, font=font)
        lines = textwrap.wrap(
            prompt, width=36
        )  # Initial wrap, adjust 'width' as needed for your font and image size
        wrapped_text = "\n".join(lines)
        left, top, right, bottom = draw.multiline_textbbox((0, 0), wrapped_text, font)
        text_width, text_height = right - left, bottom - top
        text_position = (
            (image_width - text_width) // 2,
            image_height + 10,
        )  # Center the text vertically with some padding

        # Add the text
        draw.multiline_text(
            text_position, wrapped_text, fill=text_color, font=font, align="center"
        )
        return new_image

    def post_process(
        self,
        init_image,
        mask_image,
        repainted_image,
        img_save_path,
        img_path,
        current_prompt,
        batch_index,
        count=0,
        cri_attn=None,
        new_image=None,
    ):

        concat_prompt_path=os.path.join(img_save_path,"concat_prompt")
        os.makedirs(concat_prompt_path, exist_ok=True)

        # print("cat_path:", self.concat_path)
        cat_images = []
        rewards = []
        for idx in range(len(repainted_image)):
            current_time = datetime.now()
            format_time = current_time.strftime("%Y%m%d%H%M%S.%f")[:-4]
            trans_image_path = img_path[idx]
            index = batch_index[idx]
            group_id = str(index).split('_')[1]
            name, ext = os.path.splitext(os.path.basename(trans_image_path))
            mask_image_arr = np.array(mask_image[idx].convert("L"))
            mask_image_arr = mask_image_arr[:, :, None]
            mask_image_arr = mask_image_arr.astype(np.float32) / 255.0
            #        mask_image_arr = mask_image_arr.astype("uint8")
            unmasked_unchanged_image_arr = (1 - mask_image_arr) * init_image[
                idx
            ] + mask_image_arr * repainted_image[idx]
            unmasked_unchanged_image = PIL.Image.fromarray(
                unmasked_unchanged_image_arr.round().astype("uint8")
            )
            os.makedirs(img_save_path, exist_ok=True)
            output_path = os.path.join(
                img_save_path, "{}_{}_{}{}".format(name, group_id,format_time, ext)
            )
            output_concat_prompt_path = os.path.join(
                concat_prompt_path, "{}_{}_{}{}".format(name, group_id, format_time, ext)
            )

            if new_image:
                new_image_arr = (1 - mask_image_arr) * init_image[
                    idx
                ] + mask_image_arr * new_image[idx]
                new_image_arr = PIL.Image.fromarray(
                    new_image_arr.round().astype("uint8")
                )
                new_output_path = os.path.join(
                    img_save_path, "{}_{}_{}_new{}".format(name, group_id, format_time, ext)
                )
                new_image_arr.save(new_output_path)
            else:
                new_output_path = None
            # print(output_path)
            if self.args.concat_prompt:
                concat_prompt_image = self.concat_prompt(
                    unmasked_unchanged_image, current_prompt[idx]
                )
                concat_prompt_image.save(output_concat_prompt_path)

            unmasked_unchanged_image.save(output_path)
            if self.save_concat:
                cat_image = concat_image(
                    trans_image_path,
                    output_path,
                    self.concat_path,
                    cri_attn=cri_attn,
                    new_img=new_output_path,
                )
                cat_images.append(cat_image / 1.0)
            self.log_output(
                img_path[idx],
                batch_index[idx],
                output_path,
                os.path.join(self.concat_path, os.path.basename(output_path)),
                current_prompt[idx],
            )
        return rewards

    def get_color(self):
        color_list = [
            "light pea green",
            "light yellow",
            "light pink",
            "light sky blue",
            "cream",
            "silver",
            "beige",
        ]
        color = random.choice(color_list)
        return color

    def batch_list(self, samples, batch_size):
        """Batch the given list into sublists of specified max size."""
        return [samples[i : i + batch_size] for i in range(0, len(samples), batch_size)]

    def inference_new(self, args):
        accelerator = Accelerator()
        # if args.contrast:
        #     if args.newnew_controlnet_model_path is not None:
        #         self.pipe, self.pipe_new, self.pipe_newnew = self.prepare_model(args)
        #     else:
        #         self.pipe, self.pipe_new = self.prepare_model(args)
        # else:
        self.pipe = self.prepare_model(
            args,
            device_map={"": accelerator.process_index},
        )
        print("start inference:")
        self.trans_pool()
        # image_batches = self.batch_list(self.img_list, self.batch_size)
        # prompt_batches = self.batch_list(self.prompt_list, self.batch_size)
        accelerator.wait_for_everyone()
        start = time.time()
        with torch.no_grad():
            with accelerator.split_between_processes(list(zip(self.img_list, self.prompt_list, self.index_list))) as img_prompt:
                img_prompt_batches = self.batch_list(img_prompt, self.batch_size)
                total_batches = len(img_prompt_batches)
                pbar = tqdm(total=total_batches, desc="Processing batches", disable=not accelerator.is_local_main_process)
                for batch in img_prompt_batches:
                    batch_image, batch_prompt, batch_index = zip(*batch)
                    batch_image = list(batch_image)
                    batch_prompt = list(batch_prompt)
                    batch_index = list(batch_index)

                    init_image, mask_image, edge_image = resize_and_canny(
                        batch_image,
                        self.preprocessor,
                        0.78,
                        self.width,
                        self.height,
                        self.keep_loc,
                        matting=False,
                    )

                    start = time.perf_counter()
                    generators = [
                        torch.Generator().manual_seed(42)
                        for _ in range(len(edge_image))
                    ]
                    if args.contrast:
                        new_repainted_image = self.pipe_new(
                            prompt=[current_prompt] * len(edge_image),
                            # negative_prompt=[current_config["negative_prompt"]]*len(edge_image), extra parts, distortions, additions, incorrect, uncoordinated
                            negative_prompt=[current_config["negative_prompt"]]
                            * len(edge_image),
                            generator=generators,
                            clip_skip=self.clip_skip,
                            image=edge_image,
                            guidance_scale=self.guidance_scale,
                            num_inference_steps=self.num_inference_steps,
                            width=self.width,
                            height=self.height,
                            eta=0.0,
                            controlnet_conditioning_scale=self.controlnet_conditioning_scale,
                        ).images
                    else:
                        new_repainted_image = None
                    generators = [
                        torch.Generator().manual_seed(42)
                        for _ in range(len(edge_image))
                    ]
                    repainted_image = self.pipe(
                        prompt=batch_prompt,
                        negative_prompt=[args.negative_prompt]
                        * len(
                            edge_image
                        ),  # "irregular shape, extended shape, floating, table legs, pedestal, indistinct background," "irregular shape, extended shape, floating, table legs, pedestal, improper position, improper size"
                        clip_skip=self.clip_skip,
                        generator=generators,
                        image=edge_image,
                        guidance_scale=self.guidance_scale,
                        num_inference_steps=self.num_inference_steps,
                        width=self.width,
                        height=self.height,
                        eta=0.0,
                        controlnet_conditioning_scale=self.controlnet_conditioning_scale,
                    ).images
                    end = time.perf_counter()
                    elapsed = end - start
                    # print("batch time:", elapsed)
                    if self.args.save_attn:
                        attn = []
                        net_attn_maps = get_net_attn_map(
                            repainted_image[0].size, batch_size=len(repainted_image)
                        )
                        for i in range(len(repainted_image)):
                            dir_name = "path_to_attn_maps"
                            # print(repainted_image[0].size)

                            new_attn_maps = resize_net_attn_map(
                                net_attn_maps[i], repainted_image[i].size
                            )
                            cri_attn = save_net_attn_map(
                                new_attn_maps,
                                dir_name,
                                self.pipe.tokenizer,
                                current_prompt,
                            )
                            attn.append(cri_attn)
                    #            repainted_image.save(img_save_path)
                    else:
                        attn = None
                    self.post_process(
                        init_image,
                        mask_image,
                        repainted_image,
                        self.save_path,
                        batch_image,
                        batch_prompt,
                        batch_index,
                        cri_attn=attn,
                        new_image=new_repainted_image,
                    )
                    pbar.update(1)
                pbar.close()



    def get_batch(self, num):
        return random.sample(self.img_list, num)

    def log_output(self, ori_path, unique_id, gene_path, cat_path, prompt):
        with open(self.log_file, "a") as f:
            f.writelines([ori_path, "\t", unique_id, "\t", gene_path, "\t", cat_path, "\t", prompt, '\n'])
            f.close()

    def filter_file(self, file_list):
        for idx, img in enumerate(file_list):
            file_name, ext = os.path.splitext(img)
            content = cv2.imread(img, cv2.IMREAD_UNCHANGED)
            if ext not in [".png", ".jpg"] or (content is None):
                file_list.pop(idx)
        return file_list
