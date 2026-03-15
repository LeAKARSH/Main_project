# -------------------- Dependencies --------------------
# pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install -U transformers datasets accelerate peft bitsandbytes trl huggingface_hub tqdm

import json
import torch
import os

from huggingface_hub import login

# Hardcoded Hugging Face API Key
HF_TOKEN = ""

login(token=HF_TOKEN)

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

# -------------------- Configuration --------------------

MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
DATASET_PATH = "generated_code_openrouter.jsonl"

OUTPUT_DIR = "./qwen-openscad-lora"

USE_4BIT = True

MAX_LENGTH = 512
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4

LEARNING_RATE = 2e-4
NUM_EPOCHS = 3

# -------------------------------------------------------
# Tokenizer
# -------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# -------------------------------------------------------
# Quantization (QLoRA)
# -------------------------------------------------------

if USE_4BIT:

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

else:
    bnb_config = None


# -------------------------------------------------------
# Load model
# -------------------------------------------------------

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)

# Important for QLoRA
if USE_4BIT:
    model = prepare_model_for_kbit_training(model)

# VRAM optimization
model.gradient_checkpointing_enable()
model.config.use_cache = False


# -------------------------------------------------------
# Dataset
# -------------------------------------------------------

print("📂 Loading dataset...")

data = []

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)

        text = f"""### Instruction:
Generate OpenSCAD code for: {item['prompt']}

### Response:
{item['code']}"""

        data.append({"text": text})

dataset = Dataset.from_list(data)


# -------------------------------------------------------
# Tokenization
# -------------------------------------------------------

def tokenize_function(examples):

    tokens = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

    tokens["labels"] = tokens["input_ids"].copy()

    return tokens


tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

tokenized_dataset.set_format("torch")


# -------------------------------------------------------
# LoRA config (optimized for Qwen)
# -------------------------------------------------------

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()


# -------------------------------------------------------
# Training arguments (modern settings)
# -------------------------------------------------------

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,

    num_train_epochs=NUM_EPOCHS,

    learning_rate=LEARNING_RATE,

    logging_steps=25,

    save_strategy="epoch",
    save_total_limit=2,

    optim="paged_adamw_8bit",

    bf16=torch.cuda.is_available(),

    warmup_ratio=0.03,

    lr_scheduler_type="cosine",

    gradient_checkpointing=True,

    report_to="none",

    remove_unused_columns=False
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# -------------------------------------------------------
# Train
# -------------------------------------------------------

print("🚀 Starting training...")

trainer.train()


# -------------------------------------------------------
# Save model
# -------------------------------------------------------

final_path = os.path.join(OUTPUT_DIR, "final")

model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)

print(f"✅ Model saved to {final_path}")