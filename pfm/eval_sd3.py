import os
import json
import argparse
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from diffusers import StableDiffusion3Pipeline

from pfm.diffusers_patch.pipeline_sd3 import patch_pipeline_sd3
from pfm.utils.logging import get_logger

logger = get_logger()


def setup_distributed():
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    else:
        rank, local_rank, world_size = 0, 0, 1
    device = torch.device(f"cuda:{local_rank}")
    return rank, local_rank, world_size, device


def load_prompts(path, max_samples=None):
    prompts = []
    with open(path, "r") as f:
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
            prompts.append(prompt)
    if max_samples is not None:
        prompts = prompts[:max_samples]
    return prompts


def build_pipeline(pretrained_model_path, checkpoint, device, dtype):
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        pretrained_model_path, torch_dtype=dtype,
    ).to(device)
    patch_pipeline_sd3(pipeline)

    ckpt_path = os.path.join(checkpoint, "transformer", "model.safetensors")
    state = load_file(ckpt_path)
    missing, unexpected = pipeline.transformer.load_state_dict(state, strict=False)
    if missing:
        logger.warning(f"Missing keys when loading transformer: {len(missing)} (e.g. {missing[:3]})")
    if unexpected:
        logger.warning(f"Unexpected keys when loading transformer: {len(unexpected)} (e.g. {unexpected[:3]})")

    pipeline.transformer.to(device=device, dtype=dtype)
    pipeline.transformer.eval()
    return pipeline


@torch.no_grad()
def run(args):
    rank, local_rank, world_size, device = setup_distributed()
    is_main = rank == 0
    dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16

    step_tag = Path(args.checkpoint.rstrip("/")).name

    if is_main:
        from datetime import datetime
        dt_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        dt_tag = ""
    if world_size > 1:
        obj = [dt_tag]
        dist.broadcast_object_list(obj, src=0)
        dt_tag = obj[0]

    run_root = Path(args.output_dir) / f"{step_tag}_{dt_tag}"

    prompts = load_prompts(args.val_prompts_file, args.max_samples)
    if is_main:
        logger.info(f"Loaded {len(prompts)} prompts | checkpoint={args.checkpoint}")
        logger.info(f"Sampling methods: {args.sampling_methods}")
        logger.info(f"Output root: {run_root}")

    pipeline = build_pipeline(args.pretrained_model_path, args.checkpoint, device, dtype)

    my_prompts = list(enumerate(prompts))[rank::world_size]

    for method in args.sampling_methods:
        for num_steps in args.num_steps:
            tag = f"{method}_steps{num_steps}_cfg{args.guidance_scale}"
            out_dir = run_root / tag
            if is_main:
                out_dir.mkdir(parents=True, exist_ok=True)
            if world_size > 1:
                dist.barrier()

            for global_idx, prompt in my_prompts:
                gen = torch.Generator(device=device).manual_seed(args.seed + global_idx)
                images = pipeline(
                    prompt=[prompt],
                    height=args.height,
                    width=args.width,
                    num_inference_steps=num_steps,
                    guidance_scale=args.guidance_scale,
                    generator=gen,
                    output_type="pil",
                    sampling_method=method,
                ).images
                save_path = out_dir / f"{global_idx:03d}_{prompt[:50].replace(' ', '_').replace('/', '_')}.png"
                images[0].save(str(save_path))

            if world_size > 1:
                dist.barrier()
            if is_main:
                logger.info(f"[{method} | {num_steps} steps] images saved to {out_dir}")

    if world_size > 1:
        dist.destroy_process_group()
    if is_main:
        logger.info("Evaluation complete.")


def parse_args():
    p = argparse.ArgumentParser(description="SD3 flow-matching inference / sampling ablation")
    p.add_argument("--pretrained_model_path", type=str,
                   default="stabilityai/stable-diffusion-3.5-medium")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="checkpoint dir containing transformer/model.safetensors")
    p.add_argument("--val_prompts_file", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="outputs/eval")
    p.add_argument("--sampling_methods", type=str, default="consistency,euler",
                   help="comma-separated: consistency, euler, ...")
    p.add_argument("--num_steps", type=str, default="8",
                   help="comma-separated list of inference-step counts, e.g. 4,8")
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["bf16", "fp16"])
    args = p.parse_args()
    args.sampling_methods = [m.strip() for m in args.sampling_methods.split(",") if m.strip()]
    args.num_steps = [int(s.strip()) for s in args.num_steps.split(",") if s.strip()]
    return args


if __name__ == "__main__":
    run(parse_args())
