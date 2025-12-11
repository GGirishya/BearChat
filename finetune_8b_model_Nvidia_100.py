"""
Fine-tuning Script: Llama-3.1-8B-Instruct
=========================================
Configuration:
Model: meta-llama/Llama-3.1-8B-Instruct 
Optimizations: A100 (Flash Attention 2, bfloat16)
"""

import torch
import os
import shutil
import glob
from datetime import datetime
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, SFTConfig
from google.colab import userdata #this only if you run on google collab

# Load environment variables
load_dotenv()

# ===== 1. CONFIGURATION =====
model_id = "meta-llama/Llama-3.1-8B-Instruct"
HF_TOKEN = userdata.get('HF_TOKEN')

# Checkpoint directories
CHECKPOINT_DIR = "/content/drive/MyDrive/models"
LATEST_DIR = os.path.join(CHECKPOINT_DIR, "latest")
PREVIOUS_DIR = os.path.join(CHECKPOINT_DIR, "previous")

# Interactive data file selection
print("\n" + "="*80)
print(" DATA SELECTION")
print("="*80)
DATA_FILE = input("Enter path to training data (or press Enter for latest): ").strip()
if DATA_FILE == "":
    json_files = glob.glob("msu_training_*.json") + glob.glob("Json_data_storage/*.json")
    if json_files:
        DATA_FILE = max(json_files, key=os.path.getmtime)
        print(f" Using latest file: {DATA_FILE}")
    else:
        DATA_FILE = "msu_training_data.json"
        print(f" No training files found, defaulting to: {DATA_FILE}")

# ===== 2. CHECKPOINT MANAGEMENT =====
def manage_checkpoints():
    print("\n" + "="*80)
    print(" CHECKPOINT MANAGEMENT")
    print("="*80)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    if os.path.exists(LATEST_DIR):
        print(f"\n Backing up current model...")
        if os.path.exists(PREVIOUS_DIR):
            shutil.rmtree(PREVIOUS_DIR)
        shutil.move(LATEST_DIR, PREVIOUS_DIR)
        print(f"   latest/ -> previous/ (Backup created)")
    else:
        print(f"\n No existing checkpoint to backup")

manage_checkpoints()

# ===== 3. MODEL & TOKENIZER LOADING =====
print("\n" + "="*80)
print(" LOADING MODEL (Llama 3.1 8B)")
print("="*80)

print(f"\n Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)

# --- FOR LLAMA 3.1 INFINITE GENERATION ---
tokenizer.add_special_tokens({'pad_token': '<|reserved_special_token_0|>'})
tokenizer.padding_side = 'right' # Right padding is standard for SFT
print(f"  Applied Padding Fix: Pad Token is now <|reserved_special_token_0|>")

print(f"\n Checking hardware...")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"    GPU Detected: {torch.cuda.get_device_name(0)}")
    # Enable Flash Attention 2 for A100 if available (Automatic in Transformers)
    attn_implementation = "sdpa"
else:
    device = torch.device("cpu")
    attn_implementation = "eager"

print(f"\n Loading Base Model...")
# Check for previous adapter to merge
if os.path.exists(PREVIOUS_DIR):
    print(f"   Found previous training! Loading Base + Adapter...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        token=HF_TOKEN,
        attn_implementation=attn_implementation
    )
    # Resize embeddings for the new pad token BEFORE loading adapter
    base_model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(base_model, PREVIOUS_DIR) # this will load the previous adapter and merge it with the base model
    model = model.merge_and_unload()
    print(f"   Previous knowledge merged.")
else:
    print(f"   Loading fresh base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        token=HF_TOKEN,
        attn_implementation=attn_implementation
    )
    # Resize embeddings for the new pad token
    model.resize_token_embeddings(len(tokenizer))

# Configure Pad Token ID in Model Config
model.config.pad_token_id = tokenizer.pad_token_id
print(f"  Model loaded & Embeddings resized.")

# ===== 4. LoRA CONFIGURATION (HIGH ACCURACY) =====
print("\n" + "="*80)
print(" CONFIGURING LoRA")
print("="*80)

model.gradient_checkpointing_enable()

# Target all linear layers for maximum knowledge retention
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

lora_config = LoraConfig(
    r=64,                # High Rank for knowledge
    lora_alpha=128,      # Stable scaling
    target_modules=target_modules,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ===== 5. DATASET PREPARATION =====
print("\n" + "="*80)
print(" PREPARING DATASET")
print("="*80)

if not os.path.exists(DATA_FILE):
    print(f"ERROR: File {DATA_FILE} not found.")
    exit(1)

dataset = load_dataset("json", data_files=DATA_FILE, split="train")

def format_prompt(sample):
    """
    Standard Llama 3 Format with Topic Context
    """
    # Detect Format
    if 'messages' in sample:
        # --- NEW CHAT FORMAT ---
        messages = sample['messages']
        instruction = ""
        response = ""

        for msg in messages:
            role = msg.get('role')
            content = msg.get('content', '')
            if role == 'user':
                instruction += content + "\n"
            elif role == 'assistant':
                response += content + "\n"

        instruction = instruction.strip()
        response = response.strip()

        # Topic is harder to find in Chat format, default or try to extract?
        # For now, we use the default fallback if no metadata exists
        topic = 'Missouri State University'

    else:
        # --- ORIGINAL INSTRUCTION FORMAT ---
        instruction = sample.get('instruction', '')
        response = sample.get('response', '')
        metadata = sample.get('metadata', {})
        topic = metadata.get('topic', 'Missouri State University')

    # We explicitly structure the prompt so the model learns the relationship
    # between the TOPIC and the ANSWER.
    return f"""### Topic: {topic}
### Instruction:
{instruction}

### Response:
{response}"""

# ===== 6. TRAINING ARGUMENTS (A100 OPTIMIZED) =====
print("\n" + "="*80)
print(" STARTING TRAINING")
print("="*80)

# A100 can handle batch size 16 natively for 8B model with LoRA.
# This is faster and more stable than small batches with accumulation.
training_args = SFTConfig(
    output_dir=LATEST_DIR,

    # Dataset
    dataset_text_field="text",
    max_length=4096,              # Llama 3.1 supports 128k, but 4k is plenty for training

    # A100 Optimization
    per_device_train_batch_size=16, # Native batch 16 fits on A100 40GB/80GB
    gradient_accumulation_steps=1,  # No accumulation needed if batch is 16

    # Learning Schedule (Slow & Steady for Accuracy)
    learning_rate=5e-5,
    warmup_ratio=0.1,
    num_train_epochs=3,           # 3 Epochs is usually optimal for 1k+ samples # or 7 for small 100 samples
    lr_scheduler_type="cosine",

    # Hardware
    bf16=True,                    # Mandatory for A100
    optim="adamw_torch",

    # Logging
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,

    # Accuracy Setting: No Packing
    # Ensures no bleed-over between examples
    packing=False,

    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    formatting_func=format_prompt,
    processing_class=tokenizer,
)

# Start Training
start_time = datetime.now()
trainer.train()
end_time = datetime.now()

# Save
trainer.save_model(LATEST_DIR)
tokenizer.save_pretrained(LATEST_DIR) # Save the tokenizer too!

print("\n" + "="*80)
print(f" TRAINING COMPLETE in {end_time - start_time}")
print(f" Model Saved to: {LATEST_DIR}")
print("="*80)
