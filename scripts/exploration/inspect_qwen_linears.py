import torch
from collections import Counter, defaultdict
from transformers import AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

print("Loading Model")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,
)

model.eval()

total_params = sum(p.numel() for p in model.parameters())

print()
print(f"Total parameters: {total_params:,}")
print(f"Total parameters (M): {total_params / 1e6:.2f}")
print()

linear_modules = []

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        out_features, in_features = module.weight.shape

        linear_modules.append(
            {
                "name": name,
                "in_features": in_features,
                "out_features": out_features,
                "params": module.weight.numel(),
            }
        )

print(f"Number of linear modules: {len(linear_modules)}")
print()

print("===== Linear modules =====")

for item in linear_modules:
    print(
        f"{item['name']:60s}"
        f"{item['in_features']:5d} -> {item['out_features']:5d}"
        f"params={item['params']:,}"
    )

print()
print("===== Unique linear shapes =====")

shape_counts = Counter(
    (item["in_features"], item["out_features"])
    for item in linear_modules
)

for (in_features, out_features), count in shape_counts.items():
    print(
        f"{in_features:5d} -> {out_features:5d}"
        f"count={count}"
    )