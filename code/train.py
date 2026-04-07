import os
import gc
import torch
from huggingface_hub import login
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# --- EMERGENCY VRAM HACKS ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

print("Authenticating...")
# Hardcode your token here so you don't have to type it!
login(token="token")

# Clean the VRAM slate before loading
torch.cuda.empty_cache()
gc.collect()

torch.cuda.set_per_process_memory_fraction(0.97, device=0)

# --- 1. LOAD MODEL (PURE GPU, DESKTOP MODE) ---
# Squeezing the context window to 256 tokens saves massive VRAM.
max_seq_len = 128

print("Loading model directly into the 3050...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=max_seq_len,
    dtype=torch.bfloat16,
    load_in_4bit=True,
    # No offloading limits! We are squeezing it in natively.
)

# --- 2. MICRO LORA ADAPTERS ---
model = FastLanguageModel.get_peft_model(
    model,
    r=4, # Dropped to 4. Bare minimum for style-tuning.
    target_modules=["q_proj", "v_proj"],
    lora_alpha=8,
    lora_dropout=0, 
    bias="none",    
    use_gradient_checkpointing="unsloth", 
    random_state=3407,
)

# --- 3. DATASET FORMATTING ---
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}
### Input:
{}
### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token 

def format_prompts(examples):
    texts = [
        alpaca_prompt.format(inst, inp, out) + EOS_TOKEN 
        for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"])
    ]
    return { "text" : texts }

dataset = load_dataset("json", data_files="data/data_set_fantasy.jsonl", split="train")
dataset = dataset.map(format_prompts, batched=True)

# --- 4. TRAINER (ULTRA-LOW VRAM CONFIG) ---
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_len,
    dataset_num_proc=2,
    packing=False, 
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1, # Dropped from 8 to hold fewer gradients in memory
        warmup_steps=5,
        num_train_epochs=3, 
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        logging_steps=1,
        # 🔥 THE SILVER BULLET: Paged Optimizer automatically spills to CPU RAM when full
        optim="paged_adamw_8bit", 
        weight_decay=0.01,
        output_dir="outputs",
    ),
    
)

print("Starting the training ritual...")
trainer.train()

model.save_pretrained("expand_lora_model") 
tokenizer.save_pretrained("expand_lora_model")
print("Done! Brain saved.")