import re

import torch


def merge_lora_state_dict(model_state_dict: dict, peft_model) -> dict:
    """Merge LoRA weights into base weights from a gathered state dict.

    Returns a clean state dict with base_model/lora prefixes stripped,
    where LoRA-targeted layers have merged weights: W_merged = W_base + (alpha/r) * B @ A
    """
    lora_config = peft_model.peft_config["default"]
    scaling = lora_config.lora_alpha / lora_config.r

    lora_a = {}
    lora_b = {}
    base_weights = {}
    other_weights = {}

    for key, value in model_state_dict.items():
        if "lora_A" in key:
            module_key = re.sub(r'\.lora_A\.default\.weight$', '', key)
            lora_a[module_key] = value
        elif "lora_B" in key:
            module_key = re.sub(r'\.lora_B\.default\.weight$', '', key)
            lora_b[module_key] = value
        elif ".base_layer." in key:
            module_key = re.sub(r'\.base_layer\.', '.', key)
            clean_key = re.sub(r'^base_model\.model\.', '', module_key)
            base_weights[key] = (clean_key, value)
        else:
            clean_key = re.sub(r'^base_model\.model\.', '', key)
            other_weights[clean_key] = value

    merged = dict(other_weights)
    for orig_key, (clean_key, weight) in base_weights.items():
        module_key = re.sub(r'\.base_layer\.weight$', '', orig_key)
        if module_key in lora_a and module_key in lora_b:
            a = lora_a[module_key]
            b = lora_b[module_key]
            merged[clean_key] = weight + (scaling * (b @ a)).to(weight.dtype)
        else:
            merged[clean_key] = weight

    return merged
