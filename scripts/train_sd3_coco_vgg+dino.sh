#!/bin/bash
export NCCL_DEBUG=WARN
export PYTHONPATH=$PYTHONPATH:$(pwd)
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=online

NUM_GPUS=${NUM_GPUS:-8}
EXP_NAME="sd3_coco_full_ft"
RUN_NAME="${EXP_NAME}_$(date +%Y%m%d_%H%M%S)"

torchrun --nproc_per_node=$NUM_GPUS pfm/train_sd3.py \
    --pretrained_model_path stabilityai/stable-diffusion-3-medium-diffusers \
    --dataset_type coco \
    --target_size 1024 \
    --batch_size 1 \
    --num_workers 4 \
    --learning_rate 1e-5 \
    --max_train_steps 100000 \
    --gradient_accumulation_steps 1 \
    --max_grad_norm 10.0 \
    --mixed_precision bf16 \
    --cfg_rate 0.1 \
    --weighting_scheme uniform \
    --text_max_length 512 \
    --train_flow_shift 3.0 \
    --perceptual_losses dinov2,vgg \
    --perceptual_weights 1.0,1.0 \
    --cfg_baking_scale 1.0 \
    --cfg_baking_prob 1.0 \
    --hsdp_shard_dim 8 \
    --enable_gradient_checkpointing \
    --output_dir outputs \
    --run_name $RUN_NAME \
    --log_interval 1 \
    --checkpoint_interval 500 \
    --val_interval 50 \
    --val_prompts_file evaluations/PartiPrompts.jsonl \
    --val_num_steps 8 \
    --val_guidance_scale 1.0 \
    --val_sampling_methods consistency \
    --val_height 1024 \
    --val_width 1024 \
    --val_seed 42 \
    --val_max_samples 64 \
    --use_wandb \
    --wandb_project pfm-sd3 \
    --seed 42
