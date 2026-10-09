import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,
)

model.eval()

text = "The capital of France is"
inputs = tokenizer(text, return_tensors="pt")

print("Input shape:", inputs["input_ids"].shape)

with torch.no_grad():
    outputs = model(**inputs)

print("Logits shape:", outputs.logits.shape)

next_token_id = outputs.logits[:, -1, :].argmax(dim=-1)
print("Next token:", tokenizer.decode(next_token_id))

print("Model load + forward: OK")