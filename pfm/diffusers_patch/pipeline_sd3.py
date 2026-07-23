import types
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput
from diffusers.utils.torch_utils import randn_tensor


def _calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


@torch.no_grad()
def _call_with_consistency(
    self,
    prompt=None,
    prompt_2=None,
    prompt_3=None,
    height=None,
    width=None,
    num_inference_steps=28,
    sigmas=None,
    guidance_scale=7.0,
    negative_prompt=None,
    negative_prompt_2=None,
    negative_prompt_3=None,
    num_images_per_prompt=1,
    generator=None,
    latents=None,
    prompt_embeds=None,
    negative_prompt_embeds=None,
    pooled_prompt_embeds=None,
    negative_pooled_prompt_embeds=None,
    output_type="pil",
    return_dict=True,
    joint_attention_kwargs=None,
    clip_skip=None,
    callback_on_step_end=None,
    callback_on_step_end_tensor_inputs=["latents"],
    max_sequence_length=256,
    mu=None,
    sampling_method="consistency",
    pred_type="x0",
    **kwargs,
):
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
        callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

    self.check_inputs(
        prompt, prompt_2, prompt_3, height, width,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._clip_skip = clip_skip
    self._joint_attention_kwargs = joint_attention_kwargs
    self._interrupt = False

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    lora_scale = (
        self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
    )
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = self.encode_prompt(
        prompt=prompt, prompt_2=prompt_2, prompt_3=prompt_3,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        do_classifier_free_guidance=self.do_classifier_free_guidance,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        device=device,
        clip_skip=self.clip_skip,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )

    if self.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

    num_channels_latents = self.transformer.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents, height, width,
        prompt_embeds.dtype, device, generator, latents,
    )

    scheduler_kwargs = {}
    if self.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
        _, _, h_lat, w_lat = latents.shape
        image_seq_len = (h_lat // self.transformer.config.patch_size) * (
            w_lat // self.transformer.config.patch_size
        )
        mu = _calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.16),
        )
        scheduler_kwargs["mu"] = mu
    elif mu is not None:
        scheduler_kwargs["mu"] = mu

    self.scheduler.set_timesteps(num_inference_steps, device=device, **scheduler_kwargs)
    timesteps = self.scheduler.timesteps
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    self._num_timesteps = len(timesteps)

    with self.progress_bar(total=num_inference_steps) as progress_bar:
        transformer_dtype = next(self.transformer.parameters()).dtype

        if sampling_method == "consistency":
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = latent_model_input.to(transformer_dtype)
                timestep = t.expand(latent_model_input.shape[0])

                flow_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds.to(transformer_dtype),
                    pooled_projections=pooled_prompt_embeds.to(transformer_dtype),
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    flow_pred_uncond, flow_pred_text = flow_pred.chunk(2)
                    flow_pred = flow_pred_uncond + self.guidance_scale * (flow_pred_text - flow_pred_uncond)

                sigma = (t / 1000).view(-1, 1, 1, 1).to(latents.dtype)
                image_pred = latents - flow_pred * sigma

                if i == len(timesteps) - 1:
                    progress_bar.update()
                    break

                timestep_next = timesteps[i + 1]
                sigma_next = (timestep_next / 1000).view(-1, 1, 1, 1).to(latents.dtype)
                noise_next = randn_tensor(latents.shape, generator=generator, dtype=latents.dtype, device=device)

                if pred_type == "x0":
                    latents = (1 - sigma_next) * image_pred + noise_next * sigma_next
                else:
                    dt = sigma_next - sigma
                    latents = latents + dt * flow_pred

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    pooled_prompt_embeds = callback_outputs.pop("pooled_prompt_embeds", pooled_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

            latents = image_pred
        else:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = latent_model_input.to(transformer_dtype)
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds.to(transformer_dtype),
                    pooled_projections=pooled_prompt_embeds.to(transformer_dtype),
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    pooled_prompt_embeds = callback_outputs.pop("pooled_prompt_embeds", pooled_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

    if output_type == "latent":
        image = latents
    else:
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)

    self.maybe_free_model_hooks()

    if not return_dict:
        return (image,)

    return StableDiffusion3PipelineOutput(images=image)


def patch_pipeline_sd3(pipeline):
    pipeline.__class__.__call__ = _call_with_consistency
    return pipeline
