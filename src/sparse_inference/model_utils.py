"""
Selecting, replacing and saving the Transformer Linear modules.
"""

import os

import torch


def get_target_linears(model):
    """
    Transformer Linear modules only (excludes lm_head).

    Qwen2.5-0.5B gives 168 modules: 24 layers * 7 Linear modules.
    """
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and name.startswith("model.layers.")
    }


def get_parent_module(model, module_name):
    """
    "model.layers.0.mlp.gate_proj" -> (model.layers[0].mlp, "gate_proj")
    """
    parts = module_name.split(".")

    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)

    return parent, parts[-1]


def save_model(model, tokenizer, save_path):
    if save_path is None:
        return

    save_path = os.path.expandvars(os.path.expanduser(save_path))
    os.makedirs(save_path, exist_ok=True)

    print()
    print(f"Saving pruned model to {save_path}...")

    model.save_pretrained(save_path, safe_serialization=True)
    tokenizer.save_pretrained(save_path)

    print("Save complete.")
