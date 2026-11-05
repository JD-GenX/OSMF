import os
import math
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from diffusers.utils import deprecate
from diffusers.models.attention_processor import (
    Attention,
    AttnProcessor,
    AttnProcessor2_0,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0
)


attn_maps = {}


def attn_call(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale=1.0,
    ):
    residual = hidden_states

    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)

    input_ndim = hidden_states.ndim

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )
    attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    query = attn.to_q(hidden_states, scale=scale)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

    key = attn.to_k(encoder_hidden_states, scale=scale)
    value = attn.to_v(encoder_hidden_states, scale=scale)

    query = attn.head_to_batch_dim(query)
    key = attn.head_to_batch_dim(key)
    value = attn.head_to_batch_dim(value)

    attention_probs = attn.get_attention_scores(query, key, attention_mask)

    if hasattr(self, "store_attn_map"):
        self.attn_map = attention_probs
    hidden_states = torch.bmm(attention_probs, value)
    hidden_states = attn.batch_to_head_dim(hidden_states)

    hidden_states = attn.to_out[0](hidden_states, scale=scale)
    hidden_states = attn.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if attn.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / attn.rescale_output_factor

    return hidden_states


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias.to(attn_weight.device)
    attn_weight = torch.softmax(attn_weight, dim=-1)

    return torch.dropout(attn_weight, dropout_p, train=True) @ value, attn_weight


def attn_call2_0(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale: float = 1.0,
    ):
    residual = hidden_states

    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)

    input_ndim = hidden_states.ndim

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )

    if attention_mask is not None:
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    query = attn.to_q(hidden_states, scale=scale)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

    key = attn.to_k(encoder_hidden_states, scale=scale)
    value = attn.to_v(encoder_hidden_states, scale=scale)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    if hasattr(self, "store_attn_map"):
        hidden_states, attn_map = scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        self.attn_map = attn_map
    else:
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
    hidden_states = hidden_states.to(query.dtype)

    hidden_states = attn.to_out[0](hidden_states, scale=scale)
    hidden_states = attn.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if attn.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / attn.rescale_output_factor

    return hidden_states


def lora_attn_call(self, attn: Attention, hidden_states, *args, **kwargs):
    self_cls_name = self.__class__.__name__
    deprecate(
        self_cls_name,
        "0.26.0",
        (
            f"Make sure use {self_cls_name[4:]} instead by setting"
            "LoRA layers to `self.{to_q,to_k,to_v,to_out[0]}.lora_layer` respectively. This will be done automatically when using"
            " `LoraLoaderMixin.load_lora_weights`"
        ),
    )
    attn.to_q.lora_layer = self.to_q_lora.to(hidden_states.device)
    attn.to_k.lora_layer = self.to_k_lora.to(hidden_states.device)
    attn.to_v.lora_layer = self.to_v_lora.to(hidden_states.device)
    attn.to_out[0].lora_layer = self.to_out_lora.to(hidden_states.device)

    attn._modules.pop("processor")
    attn.processor = AttnProcessor()

    if hasattr(self, "store_attn_map"):
        attn.processor.store_attn_map = True

    return attn.processor(attn, hidden_states, *args, **kwargs)


def lora_attn_call2_0(self, attn: Attention, hidden_states, *args, **kwargs):
    self_cls_name = self.__class__.__name__
    deprecate(
        self_cls_name,
        "0.26.0",
        (
            f"Make sure use {self_cls_name[4:]} instead by setting"
            "LoRA layers to `self.{to_q,to_k,to_v,to_out[0]}.lora_layer` respectively. This will be done automatically when using"
            " `LoraLoaderMixin.load_lora_weights`"
        ),
    )
    attn.to_q.lora_layer = self.to_q_lora.to(hidden_states.device)
    attn.to_k.lora_layer = self.to_k_lora.to(hidden_states.device)
    attn.to_v.lora_layer = self.to_v_lora.to(hidden_states.device)
    attn.to_out[0].lora_layer = self.to_out_lora.to(hidden_states.device)

    attn._modules.pop("processor")
    attn.processor = AttnProcessor2_0()

    if hasattr(self, "store_attn_map"):
        attn.processor.store_attn_map = True

    return attn.processor(attn, hidden_states, *args, **kwargs)


def cross_attn_init():
    AttnProcessor.__call__ = attn_call
    AttnProcessor2_0.__call__ = attn_call 
    LoRAAttnProcessor.__call__ = lora_attn_call
    LoRAAttnProcessor2_0.__call__ = lora_attn_call


def reshape_attn_map(attn_map):
    attn_map = torch.mean(attn_map,dim=0) 
    attn_map = attn_map.permute(1,0) 
    latent_size = int(math.sqrt(attn_map.shape[1]))
    latent_shape = (attn_map.shape[0],latent_size,-1)
    attn_map = attn_map.reshape(latent_shape) 

    return attn_map 


def hook_fn(name):
    def forward_hook(module, input, output):
        if hasattr(module.processor, "attn_map"):
            attn_maps[name] = module.processor.attn_map
            del module.processor.attn_map

    return forward_hook

def register_cross_attention_hook(unet):
    for name, module in unet.named_modules():
        if not name.split('.')[-1].startswith('attn2'):
            continue

        if isinstance(module.processor, AttnProcessor):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, AttnProcessor2_0):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, LoRAAttnProcessor):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, LoRAAttnProcessor2_0):
            module.processor.store_attn_map = True

        hook = module.register_forward_hook(hook_fn(name))
    
    return unet


def prompt2tokens(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    tokens = []
    for text_input_id in text_input_ids[0]:
        token = tokenizer.decoder[text_input_id.item()]
        tokens.append(token)
    return tokens


def upscale(attn_map, target_size):
    attn_map = torch.mean(attn_map, dim=0) 
    attn_map = attn_map.permute(1,0) 

    if target_size[0]*target_size[1] != attn_map.shape[1]:
        temp_size = (int(np.sqrt(attn_map.shape[1])), int(np.sqrt(attn_map.shape[1]))) 
        attn_map = attn_map.view(attn_map.shape[0], *temp_size) 
        attn_map = attn_map.unsqueeze(0) 
        attn_map = F.interpolate(
            attn_map.to(dtype=torch.float32),
            size=target_size,
            mode='bilinear',
            align_corners=False
        ).squeeze() 
    else:
        attn_map = attn_map.to(dtype=torch.float32) 

    attn_map = torch.softmax(attn_map, dim=0)
    attn_map = attn_map.reshape(attn_map.shape[0],-1) 
    return attn_map


def get_net_attn_map(image_size, batch_size=2, instance_or_negative=False, detach=True):
    target_size = (image_size[0]//8, image_size[1]//8)

    idx = 0 if instance_or_negative else 1
    total_attn_maps = []
    for i in range(batch_size):
        net_attn_maps = []

        for name, attn_map in attn_maps.items():
            if "up_blocks" in name: 
                attn_map = attn_map.cpu() if detach else attn_map

                attn_map = torch.chunk(attn_map, 2)[1] 
                attn_map = attn_map[i*8:(i+1)*8]
  
                if len(attn_map.shape) == 4:
                    attn_map = attn_map.squeeze()
                attn_map = upscale(attn_map, target_size) 
                net_attn_maps.append(attn_map) 

        net_attn_maps = torch.mean(torch.stack(net_attn_maps,dim=0),dim=0)
        net_attn_maps = net_attn_maps.reshape(net_attn_maps.shape[0], 64,64) 

        total_attn_maps.append(net_attn_maps)
    return total_attn_maps


def save_net_attn_map(net_attn_maps, dir_name, tokenizer, prompt, cri ='product'):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    tokens = prompt2tokens(tokenizer, prompt)

    total_attn_scores = 0
    for i, (token, attn_map) in enumerate(zip(tokens, net_attn_maps)):
        attn_map_score = torch.sum(attn_map)
        attn_map = attn_map.cpu().numpy()
        h,w = attn_map.shape
        attn_map_total = h*w
        attn_map_score = attn_map_score / attn_map_total
        total_attn_scores += attn_map_score

        
        token = token.replace('</w>','')
        if token == cri:
            cri_attn = attn_map

    return process_attn_map(cri_attn)


def resize_net_attn_map(net_attn_maps, target_size):
    net_attn_maps = net_attn_maps.to(dtype=torch.float32).unsqueeze(0)

    net_attn_maps = F.interpolate(
        net_attn_maps,
        size=target_size,
        mode='bilinear',
        align_corners=False
    ).squeeze(0) 
    return net_attn_maps

def process_attn_map(attn_map):
    normalized_attn_map = (attn_map - np.min(attn_map)) / (np.max(attn_map) - np.min(attn_map)) * 255
    normalized_attn_map = normalized_attn_map.astype(np.uint8)
    return normalized_attn_map

def save_attn_map(attn_map, title, save_path):
    normalized_attn_map = (attn_map - np.min(attn_map)) / (np.max(attn_map) - np.min(attn_map)) * 255
    normalized_attn_map = normalized_attn_map.astype(np.uint8)
    image = Image.fromarray(normalized_attn_map)
    image.save(save_path, format='PNG', compression=0)