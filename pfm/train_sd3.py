import os
import json
import random
import argparse
import time
from dataclasses import dataclass, field
from typing import Callable, Optional
from pathlib import Path
from collections import deque
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict
from diffusers import StableDiffusion3Pipeline
from peft import LoraConfig, get_peft_model

from pfm.dataset import (
    build_laion_dataloader,
    build_coco_dataloader,
)
from pfm.dataset.filter import check_image_filter
from pfm.dataset.registry import DATASET_CONFIGS
from pfm.diffusers_patch.pipeline_sd3 import patch_pipeline_sd3
from pfm.models.losses import PerceptualLoss
from pfm.utils.fsdp_load import maybe_load_fsdp_model
from pfm.utils.logging import get_logger, setup_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SD3TrainingConfig:
    # model
    pretrained_model_path: str = "stabilityai/stable-diffusion-3.5-medium"
    use_lora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 64
    lora_path: Optional[str] = None
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "attn.to_q", "attn.to_k", "attn.to_v",
        "attn.to_out.0",
        "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj",
        "attn.to_add_out",
        "ff.net.0.proj", "ff.net.2",
        "ff_context.net.0.proj", "ff_context.net.2",
        "proj_out",
    ])

    # dataset
    dataset_type: str = "laion"
    target_size: int = 1024
    batch_size: int = 1
    num_workers: int = 4
    shuffle: bool = True
    buffer_size: int = 1000

    # training
    learning_rate: float = 1e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_weight_decay: float = 0.0
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 10.0
    gradient_accumulation_steps: int = 1
    max_train_steps: int = 100000
    mixed_precision: str = "bf16"
    allow_tf32: bool = False
    cfg_rate: float = 0.1
    weighting_scheme: str = "uniform"
    text_max_length: int = 512
    hsdp_shard_dim: int = 8
    enable_gradient_checkpointing: bool = True
    train_flow_shift: float = 3.0
    perceptual_losses: list[str] = field(default_factory=lambda: ["vgg", "dino"])
    perceptual_weights: list[float] = field(default_factory=lambda: [1.0, 1.0])
    cfg_baking_scale: float = 1.0
    cfg_baking_prob: float = 1.0
    mse_loss_weight: float = 0.0

    # logging / checkpointing
    output_dir: str = "outputs"
    run_name: str = ""
    log_interval: int = 1
    checkpoint_interval: int = 500
    seed: int = 42
    use_wandb: bool = True
    wandb_project: str = "pfm-sd3"

    # validation
    val_interval: int = 100
    val_prompts_file: str = ""
    val_num_steps: int = 8
    val_guidance_scale: float = 1.0
    val_sampling_methods: list[str] = field(default_factory=lambda: ["consistency"])
    val_height: int = 1024
    val_width: int = 1024
    val_seed: int = 42
    val_max_samples: Optional[int] = None

    # resume
    resume_from_checkpoint: Optional[str] = None


# ---------------------------------------------------------------------------
# Timestep sampling
# ---------------------------------------------------------------------------

def sample_timesteps(batch_size, device, scheme="lognorm", shift=1.0):
    if scheme == "uniform":
        sigmas = torch.rand(batch_size, device=device)
    elif scheme == "lognorm":
        u = torch.randn(batch_size, device=device)
        sigmas = torch.sigmoid(u)
    else:
        raise ValueError(f"Unknown weighting scheme: {scheme}")

    if shift != 1.0:
        sigmas = (shift * sigmas) / (1 + (shift - 1) * sigmas)

    timesteps = (sigmas * 1000).float()
    return timesteps, sigmas


def convert_flow_pred_to_x0(flow_pred, x_t, sigma):
    if sigma.ndim != flow_pred.ndim:
        sigma = sigma.view(-1, *([1] * (flow_pred.ndim - 1))).to(flow_pred.dtype)
    return x_t - flow_pred * sigma


def decode_vae_latents(vae, latents):
    latents = latents / vae.config.scaling_factor + vae.config.shift_factor
    latents = latents.to(vae.dtype)
    pixels = vae.decode(latents, return_dict=False)[0]
    return pixels.clamp(-1, 1)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class SD3Trainer:
    def __init__(self, config: SD3TrainingConfig):
        self.config = config
        self._setup_distributed()
        self._setup_dirs()
        self._set_seed(config.seed + self.rank)
        self._build_pipeline()
        self._build_loss()
        self._build_optimizer()

        self.global_step = 0
        self.rng = random.Random(config.seed + self.rank)

        if self.is_main and config.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=config.wandb_project,
                    name=config.run_name or None,
                    config=vars(config),
                    dir=str(self.exp_dir),
                )
                self.wandb = wandb
            except ImportError:
                logger.warning("wandb not installed, disabling wandb logging")
                self.wandb = None
        else:
            self.wandb = None

    # ---- setup ----------------------------------------------------------

    def _setup_distributed(self):
        if "RANK" in os.environ:
            self.rank = int(os.environ["RANK"])
            self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group("nccl")
        else:
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1
        self.device = torch.device(f"cuda:{self.local_rank}")
        self.is_main = self.rank == 0

    def _setup_dirs(self):
        global logger
        cfg = self.config
        base = Path(cfg.output_dir) / (cfg.run_name or "default")
        self.exp_dir = base
        self.log_dir = base / "logs"
        self.ckpt_dir = base / "checkpoints"
        if self.is_main:
            for d in [self.exp_dir, self.log_dir, self.ckpt_dir]:
                d.mkdir(parents=True, exist_ok=True)
        if self.world_size > 1:
            dist.barrier()
        logger = setup_logger(str(self.log_dir))

    def _set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _build_pipeline(self):
        cfg = self.config
        dtype = torch.bfloat16 if cfg.mixed_precision == "bf16" else torch.float16

        self.pipeline = StableDiffusion3Pipeline.from_pretrained(
            cfg.pretrained_model_path, torch_dtype=dtype,
        ).to(self.device)
        patch_pipeline_sd3(self.pipeline)

        self.pipeline.vae.requires_grad_(False)
        self.pipeline.text_encoder.requires_grad_(False)
        self.pipeline.text_encoder_2.requires_grad_(False)
        if self.pipeline.text_encoder_3 is not None:
            self.pipeline.text_encoder_3.requires_grad_(False)

        transformer = self.pipeline.transformer
        if cfg.use_lora:
            transformer.requires_grad_(False)
            lora_config = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
                target_modules=cfg.lora_target_modules,
                init_lora_weights="gaussian",
                lora_dropout=0.0,
            )
            transformer = get_peft_model(transformer, lora_config)
            if cfg.lora_path is not None:
                transformer.load_adapter(cfg.lora_path, adapter_name="default")
        else:
            transformer.requires_grad_(True)

        if cfg.enable_gradient_checkpointing:
            base_model = transformer
            while hasattr(base_model, 'base_model'):
                base_model = base_model.base_model
            if hasattr(base_model, 'model'):
                base_model = base_model.model
            base_model.enable_gradient_checkpointing()

        if self.world_size > 1:
            from diffusers.models.attention import JointTransformerBlock
            transformer._fsdp_shard_conditions = [
                lambda name, module: isinstance(module, JointTransformerBlock),
            ]
            transformer = transformer.float()
            transformer = maybe_load_fsdp_model(
                transformer,
                hsdp_shard_dim=cfg.hsdp_shard_dim,
                reshard_after_forward=False,
                param_dtype=dtype,
                reduce_dtype=torch.float32,
                training_mode=True,
            )

        self.transformer = transformer
        self.pipeline.transformer = transformer

        if cfg.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        trainable = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
        if self.is_main:
            logger.info(f"Trainable parameters: {trainable / 1e6:.1f}M")

    def _build_loss(self):
        cfg = self.config
        self.perceptual_losses = []
        for net_name, weight in zip(cfg.perceptual_losses, cfg.perceptual_weights):
            pmodel = PerceptualLoss(net=net_name)
            pmodel.loss_weight = float(weight)
            pmodel.to(self.device, dtype=torch.bfloat16)
            self.perceptual_losses.append(pmodel)
        if self.is_main:
            logger.info(f"Perceptual models: {cfg.perceptual_losses} weights: {cfg.perceptual_weights}")

    def _build_optimizer(self):
        cfg = self.config
        self.optimizer = torch.optim.AdamW(
            [p for p in self.transformer.parameters() if p.requires_grad],
            lr=cfg.learning_rate,
            betas=(cfg.adam_beta1, cfg.adam_beta2),
            weight_decay=cfg.adam_weight_decay,
            eps=cfg.adam_epsilon,
        )

    # ---- dataloader -----------------------------------------------------

    def build_dataloader(self):
        cfg = self.config
        ds_cfg = DATASET_CONFIGS[cfg.dataset_type]
        ds_type = ds_cfg["type"]

        filter_fn = partial(
            check_image_filter,
            min_resolution=800,
            min_aspect_ratio=0.8,
            max_aspect_ratio=1.2,
        )
        target_size = (cfg.target_size, cfg.target_size)

        if ds_type == "laion":
            return build_laion_dataloader(
                data_dir=ds_cfg["data_dir"], split=ds_cfg["split"],
                target_size=target_size, filter_fn=filter_fn,
                batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                shuffle=cfg.shuffle, seed=cfg.seed,
            )
        elif ds_type == "coco":
            return build_coco_dataloader(
                data_dir=ds_cfg["data_dir"], split=ds_cfg["split"],
                target_size=target_size, filter_fn=filter_fn,
                batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                shuffle=cfg.shuffle, seed=cfg.seed,
            )
        else:
            raise ValueError(f"Unknown dataset type: {ds_type}")

    # ---- encoding -------------------------------------------------------

    @torch.no_grad()
    def encode_vae(self, pixels):
        pixels = pixels.to(device=self.device, dtype=self.pipeline.vae.dtype)
        latent_dist = self.pipeline.vae.encode(pixels).latent_dist
        latents = latent_dist.sample()
        latents = (latents - self.pipeline.vae.config.shift_factor) * self.pipeline.vae.config.scaling_factor
        return latents

    @torch.no_grad()
    def encode_prompt(self, captions):
        prompts = [
            ("" if self.rng.random() < self.config.cfg_rate else c)
            for c in captions
        ]
        prompt_embeds, _, pooled_prompt_embeds, _ = self.pipeline.encode_prompt(
            prompt=prompts, prompt_2=None, prompt_3=None,
            device=self.device, do_classifier_free_guidance=False,
            max_sequence_length=self.config.text_max_length,
        )

        uncond_prompts = [""] * len(captions)
        uncond_embeds, _, uncond_pooled, _ = self.pipeline.encode_prompt(
            prompt=uncond_prompts, prompt_2=None, prompt_3=None,
            device=self.device, do_classifier_free_guidance=False,
            max_sequence_length=self.config.text_max_length,
        )
        return prompt_embeds, pooled_prompt_embeds, uncond_embeds, uncond_pooled

    # ---- loss -----------------------------------------------------------

    def compute_loss(self, latents, prompt_embeds, pooled_prompt_embeds, uncond_embeds, uncond_pooled):
        cfg = self.config
        dtype = latents.dtype
        device = latents.device

        noise = torch.randn_like(latents)
        timesteps, sigmas = sample_timesteps(
            latents.shape[0], device, scheme=cfg.weighting_scheme, shift=cfg.train_flow_shift,
        )
        sigmas_reshape = sigmas.view(-1, 1, 1, 1).to(dtype)
        noisy_latents = (1 - sigmas_reshape) * latents + sigmas_reshape * noise

        model = self.transformer
        flow_cond = model(
            hidden_states=noisy_latents.to(torch.bfloat16),
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0]

        with torch.no_grad():
            flow_uncond = model(
                hidden_states=noisy_latents.to(torch.bfloat16),
                timestep=timesteps,
                encoder_hidden_states=uncond_embeds,
                pooled_projections=uncond_pooled,
                return_dict=False,
            )[0]

        guidance_scale = cfg.cfg_baking_scale
        if self.rng.random() > cfg.cfg_baking_prob:
            guidance_scale = 1.0

        if guidance_scale == 1.0:
            x0_pred = convert_flow_pred_to_x0(flow_cond, noisy_latents, sigmas_reshape)
        else:
            beta = 1.0 / guidance_scale
            flow_pred = beta * flow_cond + (1.0 - beta) * flow_uncond.detach()
            x0_pred = convert_flow_pred_to_x0(flow_pred, noisy_latents, sigmas_reshape)

        # MSE loss on velocity prediction (flow matching regression)
        velocity_target = noise - latents
        if guidance_scale == 1.0:
            velocity_pred = flow_cond
        else:
            velocity_pred = flow_pred

        x0_pred = x0_pred.to(torch.bfloat16)
        target = latents

        pred_pixels = torch.utils.checkpoint.checkpoint(
            decode_vae_latents, self.pipeline.vae, x0_pred, use_reentrant=False,
        )
        with torch.no_grad():
            target_pixels = decode_vae_latents(self.pipeline.vae, target)

        loss = torch.tensor(0.0, device=device)
        loss_details = {}
        for pmodel in self.perceptual_losses:
            pred_resized = F.interpolate(pred_pixels, size=(pmodel.input_size, pmodel.input_size), mode="bilinear", align_corners=False)
            target_resized = F.interpolate(target_pixels, size=(pmodel.input_size, pmodel.input_size), mode="bilinear", align_corners=False)
            raw_loss = pmodel(pred_resized.to(torch.bfloat16), target_resized.to(torch.bfloat16))
            weighted_loss = pmodel.loss_weight * raw_loss
            loss = loss + weighted_loss
            loss_details[pmodel.net] = weighted_loss.detach()

        if cfg.mse_loss_weight > 0.0:
            mse_loss = F.mse_loss(velocity_pred.float(), velocity_target.float())
            weighted_mse = cfg.mse_loss_weight * mse_loss
            loss = loss + weighted_mse
            loss_details["mse"] = weighted_mse.detach()

        # Normalize by adaptive weight factor (NFT-style) to remove timestep-dependent magnitude
        # with torch.no_grad():
        #     weight_factor = torch.abs(x0_pred.float() - target.float()).mean().clamp(min=1e-5)
        # loss = loss / weight_factor
        # # Normalize by sigma to counteract the v→x0 conversion coefficient
        # loss = loss / sigmas.mean().clamp(min=0.01)
        # with torch.no_grad():
        #     weight_factor = torch.abs(x0_pred.float() - target.float() - target.float()).mean().clamp(min=1e-5)
        # loss = loss / weight_factor

        loss_details["timestep"] = timesteps.mean().detach()
        # loss_details["weight_factor"] = weight_factor.detach()

        return loss, loss_details

    # ---- single step ----------------------------------------------------

    def train_step(self, batch):
        latents = self.encode_vae(batch["pixel"])
        prompt_embeds, pooled_prompt_embeds, uncond_embeds, uncond_pooled = self.encode_prompt(batch["caption"])

        self.transformer.train()
        loss, loss_details = self.compute_loss(latents, prompt_embeds, pooled_prompt_embeds, uncond_embeds, uncond_pooled)
        scaled_loss = loss / self.config.gradient_accumulation_steps
        scaled_loss.backward()

        return {"loss": loss.detach(), "loss_details": loss_details}

    # ---- validation --------------------------------------------------------

    @torch.no_grad()
    def log_validation(self, step):
        cfg = self.config
        if not cfg.val_prompts_file:
            return

        val_prompts = []
        with open(cfg.val_prompts_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                prompt = item.get("prompt", "")
                try:
                    parsed = json.loads(prompt)
                    if isinstance(parsed, dict) and "caption" in parsed:
                        prompt = parsed["caption"]
                except (json.JSONDecodeError, TypeError):
                    pass
                val_prompts.append(prompt)

        if cfg.val_max_samples is not None:
            val_prompts = val_prompts[:cfg.val_max_samples]

        my_prompts = val_prompts[self.rank::self.world_size]

        self.transformer.eval()

        for sampling_method in cfg.val_sampling_methods:
            val_dir = self.ckpt_dir.parent / "validation" / f"step_{step}_{sampling_method}"
            if self.is_main:
                val_dir.mkdir(parents=True, exist_ok=True)
            if self.world_size > 1:
                dist.barrier()

            for idx, prompt in enumerate(my_prompts):
                global_idx = idx * self.world_size + self.rank
                gen = torch.Generator(device=self.device).manual_seed(cfg.val_seed + global_idx)

                images = self.pipeline(
                    prompt=[prompt],
                    height=cfg.val_height,
                    width=cfg.val_width,
                    num_inference_steps=cfg.val_num_steps,
                    guidance_scale=cfg.val_guidance_scale,
                    generator=gen,
                    output_type="pil",
                    sampling_method=sampling_method,
                ).images

                save_path = val_dir / f"{global_idx:03d}_{prompt[:50].replace(' ', '_')}.png"
                images[0].save(str(save_path))

            if self.world_size > 1:
                dist.barrier()

            if self.is_main:
                logger.info(f"Validation images ({sampling_method}) saved at step {step}")
                if self.wandb is not None:
                    import glob as _glob
                    val_images = sorted(_glob.glob(str(val_dir / "*.png")))[:16]
                    self.wandb.log({
                        f"val/images_{sampling_method}": [self.wandb.Image(p) for p in val_images],
                    }, step=step)

        self.transformer.train()

    # ---- checkpoint -----------------------------------------------------

    def save_checkpoint(self, step):
        from torch.distributed.checkpoint.state_dict import StateDictOptions
        from peft import PeftModel
        from safetensors.torch import save_file
        from pfm.utils.checkpoint import merge_lora_state_dict

        save_dir = self.ckpt_dir / f"step_{step}"
        if self.is_main:
            save_dir.mkdir(parents=True, exist_ok=True)
            (save_dir / "transformer").mkdir(parents=True, exist_ok=True)
        if self.world_size > 1:
            dist.barrier()

        model_state = get_model_state_dict(
            self.transformer,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )

        if self.is_main:
            if isinstance(self.transformer, PeftModel):
                merged = merge_lora_state_dict(model_state, self.transformer)
                save_file(merged, str(save_dir / "transformer" / "model.safetensors"))
            else:
                save_file(model_state, str(save_dir / "transformer" / "model.safetensors"))

            torch.save({"step": step}, str(save_dir / "training_state.pt"))
            logger.info(f"Checkpoint saved at step {step} (merged weights)")

        if self.world_size > 1:
            dist.barrier()

    def load_checkpoint(self, path):
        from torch.distributed.checkpoint.state_dict import StateDictOptions
        from safetensors.torch import load_file

        state = torch.load(os.path.join(path, "training_state.pt"), map_location="cpu")
        self.global_step = state["step"]

        model_state = load_file(os.path.join(path, "transformer", "model.safetensors"))
        set_model_state_dict(
            self.transformer, model_state,
            options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
        )

        if self.world_size > 1:
            step_tensor = torch.tensor(self.global_step, device=self.device)
            dist.broadcast(step_tensor, src=0)
            self.global_step = step_tensor.item()

        if self.is_main:
            logger.info(f"Resumed from step {self.global_step}")

    # ---- main loop ------------------------------------------------------

    def train(self, dataloader):
        cfg = self.config
        if cfg.resume_from_checkpoint is not None:
            self.load_checkpoint(cfg.resume_from_checkpoint)

        if self.is_main:
            total_batch = cfg.batch_size * self.world_size * cfg.gradient_accumulation_steps
            logger.info(
                f"Training for {cfg.max_train_steps} steps | "
                f"batch={cfg.batch_size} x {self.world_size} GPUs x {cfg.gradient_accumulation_steps} accum = {total_batch}"
            )

        step_times: deque[float] = deque(maxlen=cfg.log_interval)
        data_iter = iter(dataloader)

        for step in range(self.global_step + 1, cfg.max_train_steps + 1):
            self.global_step = step
            step_start = time.perf_counter()
            self.optimizer.zero_grad()
            accum_loss = 0.0
            accum_loss_details = {}

            for accum_idx in range(cfg.gradient_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)
                metrics = self.train_step(batch)
                accum_loss += metrics["loss"].item() / cfg.gradient_accumulation_steps
                for k, v in metrics["loss_details"].items():
                    accum_loss_details[k] = accum_loss_details.get(k, 0.0) + v.item() / cfg.gradient_accumulation_steps

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.transformer.parameters(), cfg.max_grad_norm if cfg.max_grad_norm > 0 else float("inf"),
            ).item()
            self.optimizer.step()

            step_time = time.perf_counter() - step_start
            step_times.append(step_time)

            if step % cfg.log_interval == 0 and self.is_main:
                avg_time = sum(step_times) / len(step_times)
                logger.info(
                    f"Step {step}/{cfg.max_train_steps} | "
                    f"loss={accum_loss:.4f} | grad_norm={grad_norm:.4f} | "
                    f"time={step_time:.2f}s | avg={avg_time:.2f}s"
                )
                if self.wandb is not None:
                    log_dict = {
                        "loss": accum_loss,
                        "grad_norm": grad_norm,
                        "step_time": step_time,
                    }
                    for k, v in accum_loss_details.items():
                        log_dict[f"loss/{k}"] = v
                    self.wandb.log(log_dict, step=step)

            if step % cfg.checkpoint_interval == 0:
                self.save_checkpoint(step)
                if self.world_size > 1:
                    dist.barrier()

            if step == 1 or step % cfg.val_interval == 0:
                self.log_validation(step)

        self.save_checkpoint(self.global_step)
        if self.world_size > 1:
            dist.destroy_process_group()
        if self.is_main:
            logger.info("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="SD3 flow matching training")

    # model
    parser.add_argument("--pretrained_model_path", type=str,
                        default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--use_lora", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_path", type=str, default=None)

    # dataset
    parser.add_argument("--dataset_type", type=str, default="laion",
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--target_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)

    # training
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_grad_norm", type=float, default=10.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=100000)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--cfg_rate", type=float, default=0.1)
    parser.add_argument("--weighting_scheme", type=str, default="uniform",
                        choices=["uniform", "lognorm"])
    parser.add_argument("--text_max_length", type=int, default=512)
    parser.add_argument("--train_flow_shift", type=float, default=3.0)
    parser.add_argument("--perceptual_losses", type=str, default="vgg,dino")
    parser.add_argument("--perceptual_weights", type=str, default="1.0,1.0")
    parser.add_argument("--cfg_baking_scale", type=float, default=1.5)
    parser.add_argument("--cfg_baking_prob", type=float, default=1.0)
    parser.add_argument("--mse_loss_weight", type=float, default=0.0)
    parser.add_argument("--hsdp_shard_dim", type=int, default=8)
    parser.add_argument("--enable_gradient_checkpointing", action="store_true", default=True)

    # logging / checkpointing
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--checkpoint_interval", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true", default=True)
    parser.add_argument("--no_wandb", dest="use_wandb", action="store_false")
    parser.add_argument("--wandb_project", type=str, default="pfm-sd3")

    # validation
    parser.add_argument("--val_interval", type=int, default=100)
    parser.add_argument("--val_prompts_file", type=str, default="")
    parser.add_argument("--val_num_steps", type=int, default=8)
    parser.add_argument("--val_guidance_scale", type=float, default=1.0)
    parser.add_argument("--val_sampling_methods", type=str, default="consistency")
    parser.add_argument("--val_height", type=int, default=1024)
    parser.add_argument("--val_width", type=int, default=1024)
    parser.add_argument("--val_seed", type=int, default=42)
    parser.add_argument("--val_max_samples", type=int, default=None)

    # resume
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    config = SD3TrainingConfig(
        pretrained_model_path=args.pretrained_model_path,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_path=args.lora_path,
        dataset_type=args.dataset_type,
        target_size=args.target_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_train_steps=args.max_train_steps,
        mixed_precision=args.mixed_precision,
        cfg_rate=args.cfg_rate,
        weighting_scheme=args.weighting_scheme,
        text_max_length=args.text_max_length,
        train_flow_shift=args.train_flow_shift,
        perceptual_losses=args.perceptual_losses.split(","),
        perceptual_weights=[float(w) for w in args.perceptual_weights.split(",")],
        cfg_baking_scale=args.cfg_baking_scale,
        cfg_baking_prob=args.cfg_baking_prob,
        mse_loss_weight=args.mse_loss_weight,
        hsdp_shard_dim=args.hsdp_shard_dim,
        enable_gradient_checkpointing=args.enable_gradient_checkpointing,
        output_dir=args.output_dir,
        run_name=args.run_name,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
        seed=args.seed,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        val_interval=args.val_interval,
        val_prompts_file=args.val_prompts_file,
        val_num_steps=args.val_num_steps,
        val_guidance_scale=args.val_guidance_scale,
        val_sampling_methods=args.val_sampling_methods.split(","),
        val_height=args.val_height,
        val_width=args.val_width,
        val_seed=args.val_seed,
        val_max_samples=args.val_max_samples,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    trainer = SD3Trainer(config)
    dataloader = trainer.build_dataloader()
    trainer.train(dataloader)


if __name__ == "__main__":
    main()
