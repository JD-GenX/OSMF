from diffusers import StableDiffusionControlNetPipeline
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
import torch
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.models.controlnet import ControlNetModel
import copy
from torch.autograd import Variable
from diffusers.pipelines.controlnet.pipeline_controlnet import retrieve_timesteps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
class StableDiffusionControlNetPipelineExtended(StableDiffusionControlNetPipeline):
    # Return the full trajectory
    def forward_collect_traj_ddim(
        self,
        prompt = None,
        image = None,
        height = None,
        width = None,
        num_inference_steps = 50,
        guidance_scale = 7.5,
        negative_prompt = None,
        num_images_per_prompt = 1,
        eta = 1.0,
        generator = None,
        latents = None,
        prompt_embeds = None,
        negative_prompt_embeds = None,
        output_type = "pil",
        return_dict = True,
        callback = None,
        callback_steps = 1,
        cross_attention_kwargs = None,
        controlnet_conditioning_scale = 1.0,
        control_guidance_start = 0.0,
        control_guidance_end = 1.0,
        is_ddp = False,
        controlnet_copy=None,
        soft_reward=False,
        ip_adapter_image=None,
        clip_skip = None,
        guess_mode: bool = False,
        timesteps = None,
        ):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet.module if is_ddp else self.controlnet) else self.controlnet
    # 1. Check inputs. Raise error if not correct
            # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = (
                mult * [control_guidance_start],
                mult * [control_guidance_end],
            )
        # self.check_inputs(
        #         prompt,
        #         image,
        #         callback_steps,
        #         negative_prompt,
        #         prompt_embeds,
        #         negative_prompt_embeds,
        #         controlnet_conditioning_scale,
        #         control_guidance_start,
        #         control_guidance_end,
        #     )
        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
    # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        if is_ddp:
            if isinstance(controlnet.module, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
                controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.module.nets)
            
            global_pool_conditions = (
                    controlnet.module.config.global_pool_conditions
                    if isinstance(controlnet.module, ControlNetModel)
                    else controlnet.module.nets[0].config.global_pool_conditions
                )
            guess_mode = guess_mode or global_pool_conditions
        else:
            if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
                controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)
            
            global_pool_conditions = (
                    controlnet.config.global_pool_conditions
                    if isinstance(controlnet, ControlNetModel)
                    else controlnet.nets[0].config.global_pool_conditions
                )
            guess_mode = guess_mode or global_pool_conditions
    
    # here `guidance_scale` is defined analog to the guidance weight `w` of
    # equation (2) of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf .
    # `guidance_scale = 1` corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
        text_encoder_lora_scale = (
        self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=clip_skip,
        )

        if do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None:
            output_hidden_state = False if isinstance(self.unet.encoder_hid_proj, ImageProjection) else True
            image_embeds, negative_image_embeds = self.encode_image(
                ip_adapter_image, device, num_images_per_prompt, output_hidden_state
            )
            if do_classifier_free_guidance:
                image_embeds = torch.cat([negative_image_embeds, image_embeds])

        # 4. Prepare image
        if is_ddp:
            if isinstance(controlnet.module, ControlNetModel):
                image = self.prepare_image(
                    image=image,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.module.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )
                height, width = image.shape[-2:]
            elif isinstance(controlnet.module, MultiControlNetModel):
                images = []

                for image_ in image:
                    image_ = self.prepare_image(
                        image=image_,
                        width=width,
                        height=height,
                        batch_size=batch_size * num_images_per_prompt,
                        num_images_per_prompt=num_images_per_prompt,
                        device=device,
                        dtype=controlnet.module.dtype,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        guess_mode=guess_mode,
                    )

                    images.append(image_)

                image = images
                height, width = image[0].shape[-2:]
            else:
                assert False
        else:
            if isinstance(controlnet, ControlNetModel):
                image = self.prepare_image(
                    image=image,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )
                height, width = image.shape[-2:]
            elif isinstance(controlnet, MultiControlNetModel):
                images = []

                for image_ in image:
                    image_ = self.prepare_image(
                        image=image_,
                        width=width,
                        height=height,
                        batch_size=batch_size * num_images_per_prompt,
                        num_images_per_prompt=num_images_per_prompt,
                        device=device,
                        dtype=controlnet.dtype,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        guess_mode=guess_mode,
                    )

                    images.append(image_)

                image = images
                height, width = image[0].shape[-2:]
            else:
                assert False


        # 5. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        # if is_ddp:
        #     num_channels_latents = self.unet.module.in_channels
        # else:
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        # 6.5 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

    

    # 7.1 Add image embeds for IP-Adapter
    
        latents_list = []
        log_prob_list = []
        down_cont_list = []
        mid_cont_list = []
        latents_list.append(latents.detach().clone().cpu())
        # 6. Prepare extra step kwargs.
        # TODO: Logic should ideally just be moved out of the pipeline  # pylint: disable=g-bad-todo
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        added_cond_kwargs = {"image_embeds": image_embeds} if ip_adapter_image is not None else None
        controlnet_keep = []
        if is_ddp:
            for i in range(len(timesteps)):
                keeps = [
                    1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                    for s, e in zip(control_guidance_start, control_guidance_end)
                ]
                controlnet_keep.append(keeps[0] if isinstance(controlnet.module, ControlNetModel) else keeps)
        else:
            for i in range(len(timesteps)):
                keeps = [
                    1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                    for s, e in zip(control_guidance_start, control_guidance_end)
                ]
                controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)
        # 7. Denoising loop
        num_warmup_steps = (len(timesteps) - num_inference_steps * self.scheduler.order)
        is_unet_compiled = is_compiled_module(self.unet)
        is_controlnet_compiled = is_compiled_module(self.controlnet.module if is_ddp else self.controlnet)
        # is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")
        kl_list = []
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                if guess_mode and do_classifier_free_guidance:
                    # Infer ControlNet only for the conditional batch.
                    # print("true!!!!!!!!!!!!!!!!!!!!!!!")
                    control_model_input = latents
                    control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                    controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                else:
                    # print("false!!!!!!!!!!!!!!!!!!!!!!!")
                    control_model_input = latent_model_input
                    controlnet_prompt_embeds = prompt_embeds

                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]
                # print(guess_mode, cond_scale)
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    control_model_input,
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=image,
                    conditioning_scale=cond_scale,
                    guess_mode=guess_mode,
                    return_dict=False,
                )
                # for d in down_block_res_samples:
                #     print(d.size())
                for sample in down_block_res_samples:
                    # print("now:", sample.size())
                    # new_down.append(sample.detach().clone().cpu())
                    down_cont_list.append(sample.detach().clone().cpu())
                
                mid_cont_list.append(mid_block_res_sample.detach().clone().cpu())
                if guess_mode and do_classifier_free_guidance:
                    # Infered ControlNet only for the conditional batch.
                    # To apply the output of ControlNet to both the unconditional and conditional batches,
                    # add 0 to the unconditional batch to keep it unchanged.
                    down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                    mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])
                # for d in down_block_res_samples:
                #     print(d.size())
                # new_down =[]
                
                # predict the noise residual
                noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                # if soft_reward:
                #     old_noise_pred = unet_copy(
                #                 latent_model_input,
                #                 t,
                #                 encoder_hidden_states=prompt_embeds,
                #                 timestep_cond=timestep_cond,
                #                 cross_attention_kwargs=self.cross_attention_kwargs,
                #                 down_block_additional_residuals=down_block_res_samples,
                #                 mid_block_additional_residual=mid_block_res_sample,
                #                 added_cond_kwargs=added_cond_kwargs,
                #                 return_dict=False,
                #             )[0]

                #     if do_classifier_free_guidance:
                #         old_noise_pred_uncond, old_noise_pred_text = old_noise_pred.chunk(2)
                #         old_noise_pred = old_noise_pred_uncond + guidance_scale * (
                #             old_noise_pred_text - old_noise_pred_uncond
                #         )

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # now we get the predicted noise
                unsqueeze3x = lambda x: x[Ellipsis, None, None, None]
                unet_t = unsqueeze3x(torch.tensor([t])).to(noise_pred.device)
                # compute the previous noisy sample x_t -> x_t-1
                if soft_reward:
                    prev_latents = latents.clone()

                latents, log_prob = self.scheduler.step_logprob(
                    noise_pred, unet_t, latents, **extra_step_kwargs
                )
                latents = latents.prev_sample
                latents = latents.to(prompt_embeds.dtype)
                latents_list.append(latents.detach().clone().cpu())
                log_prob_list.append(log_prob.detach().clone().cpu())

                if soft_reward:
                    unet_times = unsqueeze3x(torch.tensor([t] * noise_pred.shape[0])).to(
                        noise_pred.device
                    )
                    old_log_prob = (
                        self.scheduler.step_forward_logprob(
                            noise_pred,
                            unet_times,
                            prev_latents,
                            latents,
                            **extra_step_kwargs
                        )
                        .detach()
                        .cpu()
                    )

                    kl_list.append((log_prob - old_log_prob).detach().clone().cpu())

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        # if output_type == "latent":
        #     image = latents
        # elif output_type == "pil":
        # # 8. Post-processing
        #     latents = latents.detach()
        #     latents = latents.to(prompt_embeds.dtype)
        #     image = self.decode_latents(latents)

        #     # 10. Convert to PIL
        #     image = self.numpy_to_pil(image)
        # else:
        # # 8. Post-processing
        #     image = self.decode_latents(latents)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload last model to CPU
        if (
            hasattr(self, "final_offload_hook")
            and self.final_offload_hook is not None
        ):
            self.final_offload_hook.offload()

        unconditional_prompt_embeds, guided_prompt_embeds = prompt_embeds.chunk(2)

        if soft_reward:
            kl_sum = 0
            kl_path = []
            for i in range(len(kl_list)):
                kl_sum += kl_list[i]
            kl_path.append(kl_sum.clone())
            for i in range(1, len(kl_list)):
                kl_sum -= kl_list[i - 1]
                kl_path.append(kl_sum.clone())
        else:
            kl_path = None

        return (
            image,
            latents_list,
            unconditional_prompt_embeds.detach().cpu(),
            guided_prompt_embeds.detach().cpu(),
            down_cont_list,
            mid_cont_list,
            log_prob_list,
            kl_path,
        )

  # Feed transitions pairs and old model
    def forward_calculate_logprob(
        self,
        prompt_embeds,
        latents,
        next_latents,
        down_block_res_samples,
        mid_block_res_sample,
        ts,
        image = None,
        controlnet_copy=None,
        height = None,
        width = None,
        num_inference_steps = 50,
        guidance_scale = 7.5,
        negative_prompt = None,
        num_images_per_prompt = 1,
        eta = 1.0,
        generator = None,
        negative_prompt_embeds = None,
        output_type = "pil",
        return_dict = True,
        callback = None,
        callback_steps = 1,
        cross_attention_kwargs = None,
        is_ddp = False,
        soft_reward=True,
        ip_adapter_image = None,
        timesteps = None

    ):
    # pylint: disable=line-too-long
        # added_cond_kwargs = {"image_embeds": image_embeds} if ip_adapter_image is not None else None
        # if soft_reward:
        #     unet_copy = None
    # 0. Default height and width to unet
        # if is_ddp:
        #     height = (
        #         height or self.unet.module.config.sample_size * self.vae_scale_factor
        #     )
        #     width = (
        #         width or self.unet.module.config.sample_size * self.vae_scale_factor
        #     )
        # else:
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

    # 2. Define call parameters
        device = self._execution_device
    # here `guidance_scale` is defined analog to the guidance weight `w` of
    # equation (2) of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf .
    # `guidance_scale = 1` corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        unet_times = timesteps[ts]

    # 4. Prepare latent variables
        # latents_list = []
        # latents_list.append(latents)
    # 5. Prepare extra step kwargs.
    # TODO: Logic should ideally just be moved out of the pipeline  # pylint: disable=g-bad-todo
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 6. Denoising loop
    # for training loops:
    # with self.progress_bar(total=num_inference_steps) as progress_bar:
    # for i, t in enumerate(timesteps):
    # expand the latents if we are doing classifier free guidance
        latent_model_input = (
            torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        )
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, unet_times)
        time_model_input = (
            torch.cat([unet_times] * 2)
            if do_classifier_free_guidance
            else unet_times
        )
        batch_size = int(latent_model_input.shape[0]/2)
        # print(image)
        image = self.prepare_image(
                    image=image,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=self.controlnet.module.dtype, #.module
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=False,
                )
        # height, width = image.shape[-2:]

        control_model_input = latent_model_input
        controlnet_prompt_embeds = prompt_embeds

        down_block_res_samples, mid_block_res_sample = self.controlnet(
            control_model_input,
            time_model_input,
            encoder_hidden_states=controlnet_prompt_embeds,
            controlnet_cond=image,
            conditioning_scale=1.0,
            guess_mode=False,
            return_dict=False,
                )

        if controlnet_copy is not None:
            old_down_block_res_samples, old_mid_block_res_sample = controlnet_copy(
            control_model_input,
            time_model_input,
            encoder_hidden_states=controlnet_prompt_embeds,
            controlnet_cond=image,
            conditioning_scale=1.0,
            guess_mode=False,
            return_dict=False,
                )
        
        # if  do_classifier_free_guidance:
        #             # Infered ControlNet only for the conditional batch.
        #             # To apply the output of ControlNet to both the unconditional and conditional batches,
        #             # add 0 to the unconditional batch to keep it unchanged.
        #     down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
        #     mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

    # predict the noise residual
        noise_pred = self.unet(
            latent_model_input,
            time_model_input,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            added_cond_kwargs=None,
        ).sample
        # aaa = (noise_pred)**2
    # add regularization

        if controlnet_copy is not None:
            old_noise_pred = self.unet(
                latent_model_input,
                time_model_input,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                down_block_additional_residuals=old_down_block_res_samples,
                mid_block_additional_residual=old_mid_block_res_sample,
                added_cond_kwargs=None,
            ).sample

        
    # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        if controlnet_copy is not None:
            if do_classifier_free_guidance:
                old_noise_pred_uncond, old_noise_pred_text = old_noise_pred.chunk(2)
                old_noise_pred = old_noise_pred_uncond + guidance_scale * (
                    old_noise_pred_text - old_noise_pred_uncond
                )
        # now we get the predicted noise
            kl_regularizer = (noise_pred - old_noise_pred) ** 2
        else:
            kl_regularizer = torch.zeros(noise_pred.shape[0])

        unsqueeze3x = lambda x: x[Ellipsis, None, None, None]
        unet_times = unsqueeze3x(unet_times).to(noise_pred.device)
        log_prob = self.scheduler.step_forward_logprob(
            noise_pred, unet_times, latents, next_latents, **extra_step_kwargs
        )

        return log_prob, kl_regularizer
    

    def forward_imagereward(
        self,
        prompt = None,
        image = None,
        height = None,
        width = None,
        num_inference_steps = 50,
        guidance_scale = 7.5,
        negative_prompt = None,
        num_images_per_prompt = 1,
        eta = 1.0,
        generator = None,
        latents = None,
        prompt_embeds = None,
        negative_prompt_embeds = None,
        output_type = "pil",
        return_dict = True,
        callback = None,
        callback_steps = 1,
        cross_attention_kwargs = None,
        controlnet_conditioning_scale = 1.0,
        control_guidance_start = 0.0,
        control_guidance_end = 1.0,
        is_ddp = False,
        controlnet_copy =None,
        soft_reward=True,
        ip_adapter_image=None,
        clip_skip = None,
        guess_mode: bool = False,
        timesteps = None,
        random_t = 40,
        correct_gradient = False,
        alpha = 1.0
        ):
    # 0. Default height and width to unet
        # if is_ddp:
        #     height = (
        #     height or self.unet.module.config.sample_size * self.vae_scale_factor
        # )
        #     width = (
        #     width or self.unet.module.config.sample_size * self.vae_scale_factor
        # )
        # else:
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        # controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet.module if is_ddp else self.controlnet) else self.controlnet
    # 1. Check inputs. Raise error if not correct
            # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(self.controlnet.nets) if isinstance(self.controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = (
                mult * [control_guidance_start],
                mult * [control_guidance_end],
            )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
    # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        if is_ddp:
            if isinstance(self.controlnet.module, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
                controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(self.controlnet.module.nets)
            
            global_pool_conditions = (
                    self.controlnet.module.config.global_pool_conditions
                    if isinstance(self.controlnet.module, ControlNetModel)
                    else self.controlnet.module.nets[0].config.global_pool_conditions
                )
            guess_mode = guess_mode or global_pool_conditions
        else:
            if isinstance(self.controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
                controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(self.controlnet.nets)
            
            global_pool_conditions = (
                    self.controlnet.config.global_pool_conditions
                    if isinstance(self.controlnet, ControlNetModel)
                    else self.controlnet.nets[0].config.global_pool_conditions
                )
            guess_mode = guess_mode or global_pool_conditions
    
    # here `guidance_scale` is defined analog to the guidance weight `w` of
    # equation (2) of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf .
    # `guidance_scale = 1` corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
        text_encoder_lora_scale = (
        self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=clip_skip,
        )

        if do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None:
            output_hidden_state = False if isinstance(self.unet.encoder_hid_proj, ImageProjection) else True
            image_embeds, negative_image_embeds = self.encode_image(
                ip_adapter_image, device, num_images_per_prompt, output_hidden_state
            )
            if do_classifier_free_guidance:
                image_embeds = torch.cat([negative_image_embeds, image_embeds])

        # 4. Prepare image
        if is_ddp:
            if isinstance(self.controlnet.module, ControlNetModel):
                image = self.prepare_image(
                    image=image,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=self.controlnet.module.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )
                height, width = image.shape[-2:]
            elif isinstance(self.controlnet.module, MultiControlNetModel):
                images = []

                for image_ in image:
                    image_ = self.prepare_image(
                        image=image_,
                        width=width,
                        height=height,
                        batch_size=batch_size * num_images_per_prompt,
                        num_images_per_prompt=num_images_per_prompt,
                        device=device,
                        dtype=self.controlnet.module.dtype,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        guess_mode=guess_mode,
                    )

                    images.append(image_)

                image = images
                height, width = image[0].shape[-2:]
            else:
                assert False
        else:
            if isinstance(self.controlnet, ControlNetModel):
                image = self.prepare_image(
                    image=image,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=self.controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )
                height, width = image.shape[-2:]
            elif isinstance(self.controlnet, MultiControlNetModel):
                images = []

                for image_ in image:
                    image_ = self.prepare_image(
                        image=image_,
                        width=width,
                        height=height,
                        batch_size=batch_size * num_images_per_prompt,
                        num_images_per_prompt=num_images_per_prompt,
                        device=device,
                        dtype=self.controlnet.dtype,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        guess_mode=guess_mode,
                    )

                    images.append(image_)

                image = images
                height, width = image[0].shape[-2:]
            else:
                assert False


        # 5. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        # if is_ddp:
        #     num_channels_latents = self.unet.module.in_channels
        # else:
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        # 6.5 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

    

    # 7.1 Add image embeds for IP-Adapter
    
        latents_list = []
        log_prob_list = []
        down_cont_list = []
        mid_cont_list = []
        # latents_list.append(latents.detach().clone().cpu())
        # 6. Prepare extra step kwargs.
        # TODO: Logic should ideally just be moved out of the pipeline  # pylint: disable=g-bad-todo
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        added_cond_kwargs = {"image_embeds": image_embeds} if ip_adapter_image is not None else None
        controlnet_keep = []
        if is_ddp:
            for i in range(len(timesteps)):
                keeps = [
                    1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                    for s, e in zip(control_guidance_start, control_guidance_end)
                ]
                controlnet_keep.append(keeps[0] if isinstance(self.controlnet.module, ControlNetModel) else keeps)
        else:
            for i in range(len(timesteps)):
                keeps = [
                    1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                    for s, e in zip(control_guidance_start, control_guidance_end)
                ]
                controlnet_keep.append(keeps[0] if isinstance(self.controlnet, ControlNetModel) else keeps)
        # 7. Denoising loop
        num_warmup_steps = (len(timesteps) - num_inference_steps * self.scheduler.order)
        is_unet_compiled = is_compiled_module(self.unet)
        is_controlnet_compiled = is_compiled_module(self.controlnet.module if is_ddp else self.controlnet)
        # is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")
        kl_list = []
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # with torch.autograd.detect_anomaly():#torch.no_grad():
            for i, t in enumerate(timesteps[:random_t]):
                with torch.no_grad():
            # expand the latents if we are doing classifier free guidance
                    latent_model_input = (
                        torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    )
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    if guess_mode and do_classifier_free_guidance:
                        # Infer ControlNet only for the conditional batch.
                        control_model_input = latents
                        control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                        controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                    else:
                        control_model_input = latent_model_input
                        controlnet_prompt_embeds = prompt_embeds

                    if isinstance(controlnet_keep[i], list):
                        cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                    else:
                        controlnet_cond_scale = controlnet_conditioning_scale
                        if isinstance(controlnet_cond_scale, list):
                            controlnet_cond_scale = controlnet_cond_scale[0]
                        cond_scale = controlnet_cond_scale * controlnet_keep[i]

                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        control_model_input,
                        t,
                        encoder_hidden_states=controlnet_prompt_embeds,
                        controlnet_cond=image,
                        conditioning_scale=cond_scale,
                        guess_mode=guess_mode,
                        return_dict=False,
                    )

                    if controlnet_copy is not None:
                        old_down_block_res_samples, old_mid_block_res_sample = controlnet_copy(
                        control_model_input,
                        t,
                        encoder_hidden_states=controlnet_prompt_embeds,
                        controlnet_cond=image,
                        conditioning_scale=cond_scale,
                        guess_mode=guess_mode,
                        return_dict=False,
                    )
                    
                    if guess_mode and do_classifier_free_guidance:
                        # Infered ControlNet only for the conditional batch.
                        # To apply the output of ControlNet to both the unconditional and conditional batches,
                        # add 0 to the unconditional batch to keep it unchanged.
                        down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                        mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])
                        old_down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in old_down_block_res_samples]
                        old_mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), old_mid_block_res_sample])
                    # for d in down_block_res_samples:
                    #     print(d.size())
                    # new_down =[]
                    
                    # predict the noise residual
                    noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            timestep_cond=timestep_cond,
                            cross_attention_kwargs=self.cross_attention_kwargs,
                            down_block_additional_residuals=down_block_res_samples,
                            mid_block_additional_residual=mid_block_res_sample,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )[0]

                    if soft_reward:
                        old_noise_pred = self.unet(
                                    latent_model_input,
                                    t,
                                    encoder_hidden_states=prompt_embeds,
                                    timestep_cond=timestep_cond,
                                    cross_attention_kwargs=self.cross_attention_kwargs,
                                    down_block_additional_residuals= old_down_block_res_samples,
                                    mid_block_additional_residual= old_mid_block_res_sample,
                                    added_cond_kwargs=added_cond_kwargs,
                                    return_dict=False,
                                )[0]

                        if do_classifier_free_guidance:
                            old_noise_pred_uncond, old_noise_pred_text = old_noise_pred.chunk(2)
                            old_noise_pred = old_noise_pred_uncond + guidance_scale * (
                                old_noise_pred_text - old_noise_pred_uncond
                            )

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )

                    # now we get the predicted noise
                    # unsqueeze3x = lambda x: x[Ellipsis, None, None, None]
                    # unet_t = unsqueeze3x(torch.tensor([t])).to(noise_pred.device)
                    # compute the previous noisy sample x_t -> x_t-1
                    # if soft_reward:
                    #     prev_latents = latents.clone()

                    # latents, log_prob = self.scheduler.step_logprob(  #
                    #     noise_pred, unet_t, latents, **extra_step_kwargs
                    # )
                    # print(t)
                    latents = self.scheduler.step(  #
                        noise_pred, t, latents, **extra_step_kwargs
                    )
                    
                    latents = latents.prev_sample
                    latents = latents.to(prompt_embeds.dtype)
                    # if i == len(timesteps) - 1 or (
                    #     (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                    # ):
                    #     progress_bar.update()
                    #     if callback is not None and i % callback_steps == 0:
                    #         step_idx = i // getattr(self.scheduler, "order", 1)
                    #         callback(step_idx, t, latents)

            # last_step (random_t)
            latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timesteps[random_t])

            if guess_mode and do_classifier_free_guidance:
                # Infer ControlNet only for the conditional batch.
                control_model_input = latents
                control_model_input = self.scheduler.scale_model_input(control_model_input, timesteps[random_t])
                controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
            else:
                control_model_input = latent_model_input
                controlnet_prompt_embeds = prompt_embeds

            if isinstance(controlnet_keep[i], list):
                cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
            else:
                controlnet_cond_scale = controlnet_conditioning_scale
                if isinstance(controlnet_cond_scale, list):
                    controlnet_cond_scale = controlnet_cond_scale[0]
                cond_scale = controlnet_cond_scale * controlnet_keep[i]

            down_block_res_samples, mid_block_res_sample = self.controlnet(
                control_model_input,
                timesteps[random_t],
                encoder_hidden_states=controlnet_prompt_embeds,
                controlnet_cond=image,
                conditioning_scale=cond_scale,
                guess_mode=guess_mode,
                return_dict=False,
            )

            if controlnet_copy is not None:
                old_down_block_res_samples, old_mid_block_res_sample = controlnet_copy(
                control_model_input,
                t,
                encoder_hidden_states=controlnet_prompt_embeds,
                controlnet_cond=image,
                conditioning_scale=cond_scale,
                guess_mode=guess_mode,
                return_dict=False,
                )
            # for d in down_block_res_samples:
            #     print(d.size())
            # for sample in down_block_res_samples:
            #     # print("now:", sample.size())
            #     # new_down.append(sample.detach().clone().cpu())
            #     down_cont_list.append(sample.detach().clone().cpu())
            
            # mid_cont_list.append(mid_block_res_sample.detach().clone().cpu())
            if guess_mode and do_classifier_free_guidance:
                # Infered ControlNet only for the conditional batch.
                # To apply the output of ControlNet to both the unconditional and conditional batches,
                # add 0 to the unconditional batch to keep it unchanged.
                down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])
                old_down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in old_down_block_res_samples]
                old_mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), old_mid_block_res_sample])
            # for d in down_block_res_samples:
            #     print(d.size())
            # new_down =[]
            
            # predict the noise residual
            noise_pred = self.unet(
                    latent_model_input,
                    timesteps[random_t],
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

            if soft_reward:
                old_noise_pred = self.unet(
                            latent_model_input,
                            timesteps[random_t],
                            encoder_hidden_states=prompt_embeds,
                            timestep_cond=timestep_cond,
                            cross_attention_kwargs=self.cross_attention_kwargs,
                            down_block_additional_residuals=old_down_block_res_samples,
                            mid_block_additional_residual=old_mid_block_res_sample,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )[0]

                if do_classifier_free_guidance:
                    old_noise_pred_uncond, old_noise_pred_text = old_noise_pred.chunk(2)
                    old_noise_pred = old_noise_pred_uncond + guidance_scale * (
                        old_noise_pred_text - old_noise_pred_uncond
                    )

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            if correct_gradient:
                def corr_func(grad):
                    bt_size = grad.size()[0]
                    alpha_prod_t = self.scheduler.alphas_cumprod[timesteps[random_t]].to(grad.device)
                    pre = (- 1.0/((1.0-alpha_prod_t) ** (0.5))).to(grad.device)
                    para = Variable(copy.deepcopy(pre * guidance_scale * (noise_pred_text - noise_pred_uncond)).detach()).type_as(grad)

                    dot_products = torch.einsum('bijk,bijk->b', [grad.clone().detach(), para])

# Compute the norms of grad2 for each batch
                    norms_grad = torch.einsum('bijk,bijk->b', [para, para])
                    print( str(alpha)+"pre: ",  alpha*(dot_products / norms_grad).mean())
                    # print((torch.matmul(grad.clone().detach().view(bt_size, -1), para.view(bt_size, -1)) / (torch.matmul(para.view(bt_size, -1), para.view(bt_size, -1))+ 1e-12)).size())
                    # print(torch.matmul(grad, para) / torch.matmul(para, para))
                    # print((torch.matmul(grad, para) / torch.matmul(para, para)).size(), grad.size(), para.size(), torch.min(grad - Variable(torch.mul(torch.mul(grad.clone().detach(), para) / torch.mul(para, para) , para)).type_as(grad)), torch.max(grad - Variable(torch.mul(torch.mul(grad.clone().detach(), para) / torch.mul(para, para) , para)).type_as(grad)), torch.min(torch.mul(para, para)), torch.max(torch.mul(para, para)))
                    
                    # return grad - Variable(torch.mul(torch.mul(grad.clone().detach(), para) / (torch.mul(para, para)+ 1e-12) , para)).type_as(grad).clone().detach()
                    # return grad - Variable((torch.matmul(grad.clone().detach().view(bt_size, -1), para.view(bt_size, -1)) / (torch.matmul(para.view(bt_size, -1), para.view(bt_size, -1))+ 1e-12) * para)).type_as(grad).clone().detach()
                    return grad - Variable(  alpha * (dot_products / norms_grad).view(-1, 1, 1, 1) * para ).type_as(grad).clone().detach()
                h = noise_pred.register_hook(corr_func)

            # now we get the predicted noise
            unsqueeze3x = lambda x: x[Ellipsis, None, None, None]
            unet_t = unsqueeze3x(torch.tensor([timesteps[random_t]])).to(noise_pred.device)
            pred_ori = self.scheduler.step(noise_pred, timesteps[random_t], latents).pred_original_sample

        if soft_reward:
            # kl_regularizer = (noise_pred - old_noise_pred) ** 2
            kl_regularizer = ((noise_pred_text - noise_pred_uncond) - (old_noise_pred_text - old_noise_pred_uncond)) ** 2
        else:
            kl_regularizer = 0.0
        if not output_type == "latent":
            image = self.vae.decode(pred_ori / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
            # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = pred_ori
            has_nsfw_concept = None

        # if has_nsfw_concept is None:
        #     do_denormalize = [True] * image.shape[0]
        # else:
        #     do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        # latents = 1 / self.vae.config.scaling_factor * latents
        # image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        # print(image.size())# image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = image.transpose(1,3)
        # print(image.size())
        image = image.transpose(1,2)
        #image = image.transpose(1,3) #(image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1)
        # print(image.size())
        # image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)  #bug!!!

        # Offload last model to CPU
        if (
            hasattr(self, "final_offload_hook")
            and self.final_offload_hook is not None
        ):
            self.final_offload_hook.offload()

        # unconditional_prompt_embeds, guided_prompt_embeds = prompt_embeds.chunk(2)
        if correct_gradient:
            return image, kl_regularizer, h
        return image, kl_regularizer, None




    def forward_dpo(
        self,
        prompt = None,
        image = None,
        height = None,
        width = None,
        num_inference_steps = 40,
        guidance_scale = 7.5,
        negative_prompt = None,
        num_images_per_prompt = 1,
        eta = 0.0,
        generator = None,
        latents = None,
        prompt_embeds = None,
        negative_prompt_embeds = None,
        output_type = "pil",
        return_dict = True,
        callback = None,
        callback_steps = 1,
        cross_attention_kwargs = None,
        controlnet_conditioning_scale = 1.0,
        control_guidance_start = 0.0,
        control_guidance_end = 1.0,
        is_ddp = False,
        controlnet_copy=None,
        soft_reward=False,
        ip_adapter_image=None,
        clip_skip = None,
        guess_mode: bool = False,
        timesteps = None,
        ):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet.module if is_ddp else self.controlnet) else self.controlnet
    # 1. Check inputs. Raise error if not correct
            # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = (
                mult * [control_guidance_start],
                mult * [control_guidance_end],
            )
        # self.check_inputs(
        #         prompt,
        #         image,
        #         callback_steps,
        #         negative_prompt,
        #         prompt_embeds,
        #         negative_prompt_embeds,
        #         controlnet_conditioning_scale,
        #         control_guidance_start,
        #         control_guidance_end,
        #     )
        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
    # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        if is_ddp:
            if isinstance(controlnet.module, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
                controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.module.nets)
            
            global_pool_conditions = (
                    controlnet.module.config.global_pool_conditions
                    if isinstance(controlnet.module, ControlNetModel)
                    else controlnet.module.nets[0].config.global_pool_conditions
                )
            guess_mode = guess_mode or global_pool_conditions
        else:
            if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
                controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)
            
            global_pool_conditions = (
                    controlnet.config.global_pool_conditions
                    if isinstance(controlnet, ControlNetModel)
                    else controlnet.nets[0].config.global_pool_conditions
                )
            guess_mode = guess_mode or global_pool_conditions
    
    # here `guidance_scale` is defined analog to the guidance weight `w` of
    # equation (2) of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf .
    # `guidance_scale = 1` corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
        text_encoder_lora_scale = (
        self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=clip_skip,
        )

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None:
            output_hidden_state = False if isinstance(self.unet.encoder_hid_proj, ImageProjection) else True
            image_embeds, negative_image_embeds = self.encode_image(
                ip_adapter_image, device, num_images_per_prompt, output_hidden_state
            )
            if do_classifier_free_guidance:
                image_embeds = torch.cat([negative_image_embeds, image_embeds])

        # 4. Prepare image
        if is_ddp:
            if isinstance(controlnet.module, ControlNetModel):
                image = self.prepare_image(
                    image=image,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.module.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )
                height, width = image.shape[-2:]
            elif isinstance(controlnet.module, MultiControlNetModel):
                images = []

                for image_ in image:
                    image_ = self.prepare_image(
                        image=image_,
                        width=width,
                        height=height,
                        batch_size=batch_size * num_images_per_prompt,
                        num_images_per_prompt=num_images_per_prompt,
                        device=device,
                        dtype=controlnet.module.dtype,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        guess_mode=guess_mode,
                    )

                    images.append(image_)

                image = images
                height, width = image[0].shape[-2:]
            else:
                assert False
        else:
            if isinstance(controlnet, ControlNetModel):
                image = self.prepare_image(
                    image=image,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )
                height, width = image.shape[-2:]
            elif isinstance(controlnet, MultiControlNetModel):
                images = []

                for image_ in image:
                    image_ = self.prepare_image(
                        image=image_,
                        width=width,
                        height=height,
                        batch_size=batch_size * num_images_per_prompt,
                        num_images_per_prompt=num_images_per_prompt,
                        device=device,
                        dtype=controlnet.dtype,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        guess_mode=guess_mode,
                    )

                    images.append(image_)

                image = images
                height, width = image[0].shape[-2:]
            else:
                assert False


        # 5. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        # if is_ddp:
        #     num_channels_latents = self.unet.module.in_channels
        # else:
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        # 6.5 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

    

    # 7.1 Add image embeds for IP-Adapter
    
        latents_list = []
        log_prob_list = []
        down_cont_list = []
        mid_cont_list = []
        latents_list.append(latents.detach().clone().cpu())
        # 6. Prepare extra step kwargs.
        # TODO: Logic should ideally just be moved out of the pipeline  # pylint: disable=g-bad-todo
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        added_cond_kwargs = {"image_embeds": image_embeds} if ip_adapter_image is not None else None
        controlnet_keep = []
        if is_ddp:
            for i in range(len(timesteps)):
                keeps = [
                    1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                    for s, e in zip(control_guidance_start, control_guidance_end)
                ]
                controlnet_keep.append(keeps[0] if isinstance(controlnet.module, ControlNetModel) else keeps)
        else:
            for i in range(len(timesteps)):
                keeps = [
                    1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                    for s, e in zip(control_guidance_start, control_guidance_end)
                ]
                controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)
        # 7. Denoising loop
        num_warmup_steps = (len(timesteps) - num_inference_steps * self.scheduler.order)
        is_unet_compiled = is_compiled_module(self.unet)
        is_controlnet_compiled = is_compiled_module(self.controlnet.module if is_ddp else self.controlnet)
        # is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")
        kl_list = []
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                if guess_mode and do_classifier_free_guidance:
                    # Infer ControlNet only for the conditional batch.
                    # print("true!!!!!!!!!!!!!!!!!!!!!!!")
                    control_model_input = latents
                    control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                    controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                else:
                    # print("false!!!!!!!!!!!!!!!!!!!!!!!")
                    control_model_input = latent_model_input
                    controlnet_prompt_embeds = prompt_embeds

                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]
                # print(guess_mode, cond_scale)
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    control_model_input,
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=image,
                    conditioning_scale=cond_scale,
                    guess_mode=guess_mode,
                    return_dict=False,
                )
                # for d in down_block_res_samples:
                #     print(d.size())
                # for sample in down_block_res_samples:
                #     # print("now:", sample.size())
                #     # new_down.append(sample.detach().clone().cpu())
                #     down_cont_list.append(sample.detach().clone().cpu())
                
                # mid_cont_list.append(mid_block_res_sample.detach().clone().cpu())
                if guess_mode and do_classifier_free_guidance:
                    # Infered ControlNet only for the conditional batch.
                    # To apply the output of ControlNet to both the unconditional and conditional batches,
                    # add 0 to the unconditional batch to keep it unchanged.
                    down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                    mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])
                # for d in down_block_res_samples:
                #     print(d.size())
                # new_down =[]
                
                # predict the noise residual
                noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # now we get the predicted noise
                unsqueeze3x = lambda x: x[Ellipsis, None, None, None]
                unet_t = unsqueeze3x(torch.tensor([t])).to(noise_pred.device)
                # compute the previous noisy sample x_t -> x_t-1
                if soft_reward:
                    prev_latents = latents.clone()

                latents, log_prob = self.scheduler.ddim_step_with_logprob(
                    noise_pred, t, latents, **extra_step_kwargs
                )
                # latents = latents.prev_sample
                latents = latents.to(prompt_embeds.dtype)
                latents_list.append(latents.detach().clone().cpu())
                log_prob_list.append(log_prob.detach().clone().cpu())

                if soft_reward:
                    unet_times = unsqueeze3x(torch.tensor([t] * noise_pred.shape[0])).to(
                        noise_pred.device
                    )
                    old_log_prob = (
                        self.scheduler.ddim_step_with_logprob(
                            noise_pred,
                            unet_times,
                            prev_latents,
                            latents,
                            **extra_step_kwargs
                        )
                        .detach()
                        .cpu()
                    )

                    kl_list.append((log_prob - old_log_prob).detach().clone().cpu())

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        # if output_type == "latent":
        #     image = latents
        # elif output_type == "pil":
        # # 8. Post-processing
        #     latents = latents.detach()
        #     latents = latents.to(prompt_embeds.dtype)
        #     image = self.decode_latents(latents)

        #     # 10. Convert to PIL
        #     image = self.numpy_to_pil(image)
        # else:
        # # 8. Post-processing
        #     image = self.decode_latents(latents)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload last model to CPU
        if (
            hasattr(self, "final_offload_hook")
            and self.final_offload_hook is not None
        ):
            self.final_offload_hook.offload()

        unconditional_prompt_embeds, guided_prompt_embeds = prompt_embeds.chunk(2)

        if soft_reward:
            kl_sum = 0
            kl_path = []
            for i in range(len(kl_list)):
                kl_sum += kl_list[i]
            kl_path.append(kl_sum.clone())
            for i in range(1, len(kl_list)):
                kl_sum -= kl_list[i - 1]
                kl_path.append(kl_sum.clone())
        else:
            kl_path = None

        return (
            image,
            latents_list,
            unconditional_prompt_embeds.detach().cpu(),
            guided_prompt_embeds.detach().cpu(),
            log_prob_list,
            kl_path,
        )
    

    def forward_dpo_calculate_logprob(
        self,
        prompt_embeds,
        latents,
        next_latents,
        ts,
        image = None,
        controlnet_copy=None,
        height = None,
        width = None,
        num_inference_steps = 50,
        guidance_scale = 7.5,
        negative_prompt = None,
        num_images_per_prompt = 1,
        eta = 1.0,
        generator = None,
        negative_prompt_embeds = None,
        output_type = "pil",
        return_dict = True,
        callback = None,
        callback_steps = 1,
        cross_attention_kwargs = None,
        is_ddp = False,
        soft_reward=True,
        ip_adapter_image = None,
        timesteps = None

    ):
    # pylint: disable=line-too-long
        # added_cond_kwargs = {"image_embeds": image_embeds} if ip_adapter_image is not None else None
        # if soft_reward:
        #     unet_copy = None
    # 0. Default height and width to unet
        # if is_ddp:
        #     height = (
        #         height or self.unet.module.config.sample_size * self.vae_scale_factor
        #     )
        #     width = (
        #         width or self.unet.module.config.sample_size * self.vae_scale_factor
        #     )
        # else:
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

    # 2. Define call parameters
        device = self._execution_device
    # here `guidance_scale` is defined analog to the guidance weight `w` of
    # equation (2) of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf .
    # `guidance_scale = 1` corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        unet_times = timesteps[ts]

    # 4. Prepare latent variables
        # latents_list = []
        # latents_list.append(latents)
    # 5. Prepare extra step kwargs.
    # TODO: Logic should ideally just be moved out of the pipeline  # pylint: disable=g-bad-todo
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 6. Denoising loop
    # for training loops:
    # with self.progress_bar(total=num_inference_steps) as progress_bar:
    # for i, t in enumerate(timesteps):
    # expand the latents if we are doing classifier free guidance
        latent_model_input = (
            torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        )
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, unet_times)
        time_model_input = (
            torch.cat([unet_times] * 2)
            if do_classifier_free_guidance
            else unet_times
        )
        batch_size = int(latent_model_input.shape[0]/2)
        
        image = self.prepare_image(
                    image=image,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=self.controlnet.module.dtype, #.module
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=False,
                )
        # height, width = image.shape[-2:]
        # print(len(image),latent_model_input.size(), prompt_embeds.size())
        control_model_input = latent_model_input
        controlnet_prompt_embeds = prompt_embeds

        down_block_res_samples, mid_block_res_sample = self.controlnet(
            control_model_input,
            time_model_input,
            encoder_hidden_states=controlnet_prompt_embeds,
            controlnet_cond=image,
            conditioning_scale=1.0,
            guess_mode=False,
            return_dict=False,
                )

        if controlnet_copy is not None:
            old_down_block_res_samples, old_mid_block_res_sample = controlnet_copy(
            control_model_input,
            time_model_input,
            encoder_hidden_states=controlnet_prompt_embeds,
            controlnet_cond=image,
            conditioning_scale=1.0,
            guess_mode=False,
            return_dict=False,
                )
        
        # if  do_classifier_free_guidance:
        #             # Infered ControlNet only for the conditional batch.
        #             # To apply the output of ControlNet to both the unconditional and conditional batches,
        #             # add 0 to the unconditional batch to keep it unchanged.
        #     down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
        #     mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

    # predict the noise residual
        noise_pred = self.unet(
            latent_model_input,
            time_model_input,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            added_cond_kwargs=None,
        ).sample
        # aaa = (noise_pred)**2
    # add regularization

        if controlnet_copy is not None:
            old_noise_pred = self.unet(
                latent_model_input,
                time_model_input,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                down_block_additional_residuals=old_down_block_res_samples,
                mid_block_additional_residual=old_mid_block_res_sample,
                added_cond_kwargs=None,
            ).sample

        
    # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        if controlnet_copy is not None:
            if do_classifier_free_guidance:
                old_noise_pred_uncond, old_noise_pred_text = old_noise_pred.chunk(2)
                old_noise_pred = old_noise_pred_uncond + guidance_scale * (
                    old_noise_pred_text - old_noise_pred_uncond
                )
        # now we get the predicted noise
            kl_regularizer = (noise_pred - old_noise_pred) ** 2
        else:
            kl_regularizer = torch.zeros(noise_pred.shape[0])

        unsqueeze3x = lambda x: x[Ellipsis, None, None, None]
        unet_times = unsqueeze3x(unet_times).to(noise_pred.device)
        log_prob = self.scheduler.step_forward_logprob(
            noise_pred, unet_times, latents, next_latents, **extra_step_kwargs
        )
        old_log_prob = self.scheduler.step_forward_logprob(
            old_noise_pred, unet_times, latents, next_latents, **extra_step_kwargs
        )

        return log_prob, old_log_prob, kl_regularizer
    

    @torch.no_grad()
    # @replace_example_docstring(EXAMPLE_DOC_STRING)
    def single_sample(
        self,
        prompt: Union[str, List[str]] = None,
        image = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        step_images = False, 
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.FloatTensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The ControlNet input condition to provide guidance to the `unet` for generation. If the type is
                specified as `torch.FloatTensor`, it is passed to ControlNet as is. `PIL.Image.Image` can also be
                accepted as an image. The dimensions of the output image defaults to `image`'s dimensions. If height
                and/or width are passed, `image` is resized accordingly. If multiple ControlNets are specified in
                `init`, images must be passed as a list such that each element of the list can be correctly batched for
                input to a single ControlNet.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original `unet`. If multiple ControlNets are specified in `init`, you can set
                the corresponding scale as a list.
            guess_mode (`bool`, *optional*, defaults to `False`):
                The ControlNet encoder tries to recognize the content of the input image even if you remove all
                prompts. A `guidance_scale` value between 3.0 and 5.0 is recommended.
            control_guidance_start (`float` or `List[float]`, *optional*, defaults to 0.0):
                The percentage of total steps at which the ControlNet starts applying.
            control_guidance_end (`float` or `List[float]`, *optional*, defaults to 1.0):
                The percentage of total steps at which the ControlNet stops applying.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeine class.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        # if callback is not None:
        #     deprecate(
        #         "callback",
        #         "1.0.0",
        #         "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
        #     )
        # if callback_steps is not None:
        #     deprecate(
        #         "callback_steps",
        #         "1.0.0",
        #         "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
        #     )

        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = (
                mult * [control_guidance_start],
                mult * [control_guidance_end],
            )

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            image,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            controlnet_conditioning_scale,
            control_guidance_start,
            control_guidance_end,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = guess_mode or global_pool_conditions

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None:
            output_hidden_state = False if isinstance(self.unet.encoder_hid_proj, ImageProjection) else True
            image_embeds, negative_image_embeds = self.encode_image(
                ip_adapter_image, device, num_images_per_prompt, output_hidden_state
            )
            if self.do_classifier_free_guidance:
                image_embeds = torch.cat([negative_image_embeds, image_embeds])

        # 4. Prepare image
        if isinstance(controlnet, ControlNetModel):
            image = self.prepare_image(
                image=image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                guess_mode=guess_mode,
            )
            height, width = image.shape[-2:]
        elif isinstance(controlnet, MultiControlNetModel):
            images = []

            for image_ in image:
                image_ = self.prepare_image(
                    image=image_,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )

                images.append(image_)

            image = images
            height, width = image[0].shape[-2:]
        else:
            assert False

        # 5. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)

        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6.5 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Add image embeds for IP-Adapter
        added_cond_kwargs = {"image_embeds": image_embeds} if ip_adapter_image is not None else None

        # 7.2 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        is_unet_compiled = is_compiled_module(self.unet)
        is_controlnet_compiled = is_compiled_module(self.controlnet)
        is_torch_higher_equal_2_1 = False
        step_image = []
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Relevant thread:
                # https://dev-discuss.pytorch.org/t/cudagraphs-in-pytorch-2-0/1428
                if (is_unet_compiled and is_controlnet_compiled) and is_torch_higher_equal_2_1:
                    torch._inductor.cudagraph_mark_step_begin()
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # controlnet(s) inference
                if guess_mode and self.do_classifier_free_guidance:
                    # Infer ControlNet only for the conditional batch.
                    control_model_input = latents
                    control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                    controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                else:
                    control_model_input = latent_model_input
                    controlnet_prompt_embeds = prompt_embeds

                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]

                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    control_model_input,
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=image,
                    conditioning_scale=cond_scale,
                    guess_mode=guess_mode,
                    return_dict=False,
                )

                if guess_mode and self.do_classifier_free_guidance:
                    # Infered ControlNet only for the conditional batch.
                    # To apply the output of ControlNet to both the unconditional and conditional batches,
                    # add 0 to the unconditional batch to keep it unchanged.
                    down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                    mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if step_images:
                    cur_image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
                    cur_image = self.image_processor.postprocess(cur_image, output_type=output_type, do_denormalize=None)
                    step_image.append(cur_image)
                
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.to("cpu")
            torch.cuda.empty_cache()

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        # if not return_dict:
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept), step_image
            
