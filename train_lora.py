"""
LoRA Fine-tuning for Flan-T5 on Indian Legal Documents
Dataset: indian_legal_dataset.jsonl (instruction, input, output format)
Model: google/flan-t5-base
Hardware: CPU with 8GB RAM
"""

import json
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
import time

# ─── CONFIG ───────────────────────────────────────────────────────────────────
JSONL_FILES = [
    "/home/tharani/Downloads/indian_legal_dataset.jsonl",
    "/home/tharani/Downloads/indian_legal_dataset (1).jsonl",
]
MODEL_NAME      = "google/flan-t5-base"
OUTPUT_DIR      = "/home/tharani/justact_bert/lora_model"
MAX_INPUT_LEN   = 512
MAX_TARGET_LEN  = 200
BATCH_SIZE      = 2        # small for CPU
GRAD_ACCUM      = 8        # effective batch = 16
EPOCHS          = 3
LR              = 3e-4
MAX_SAMPLES     = 500      # use 500 examples — enough for good results, fast on CPU
LORA_R          = 8
LORA_ALPHA      = 16
LORA_DROPOUT    = 0.05
SAVE_EVERY      = 100      # save checkpoint every N steps
SEED            = 42
# ──────────────────────────────────────────────────────────────────────────────

random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 60)
print("  LoRA Fine-tuning: Flan-T5 on Indian Legal Documents")
print("=" * 60)


# ─── LOAD DATASET ─────────────────────────────────────────────────────────────

def load_jsonl(paths, max_samples=None):
    """Load and merge JSONL files, shuffle, and limit samples."""
    data = []
    for path in paths:
        if not os.path.exists(path):
            print(f"[WARN] File not found: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    # Must have input and output
                    if obj.get("input") and obj.get("output"):
                        data.append(obj)
                except json.JSONDecodeError:
                    continue
        print(f"[DATA] Loaded from {os.path.basename(path)}: {len(data)} total so far")

    # Shuffle
    random.shuffle(data)

    # Limit
    if max_samples and len(data) > max_samples:
        data = data[:max_samples]
        print(f"[DATA] Limited to {max_samples} samples")

    print(f"[DATA] Final dataset size: {len(data)} examples")
    return data


def make_prompt(item):
    """Convert JSONL item to T5 input prompt."""
    instruction = item.get("instruction", "Summarize the following Indian legal case document.")
    inp = item.get("input", "")
    # Truncate input to avoid token limit issues
    words = inp.split()
    if len(words) > 400:
        inp = " ".join(words[:400]) + "..."
    return f"{instruction}\n\n{inp}"


# ─── DATASET CLASS ────────────────────────────────────────────────────────────

class LegalDataset(Dataset):
    def __init__(self, data, tokenizer, max_input_len, max_target_len):
        self.data           = data
        self.tokenizer      = tokenizer
        self.max_input_len  = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item   = self.data[idx]
        prompt = make_prompt(item)
        target = item.get("output", "")

        # Truncate target
        target_words = target.split()
        if len(target_words) > 150:
            target = " ".join(target_words[:150])

        # Tokenize input
        enc = self.tokenizer(
            prompt,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize target
        dec = self.tokenizer(
            text_target=target,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = dec["input_ids"].squeeze()
        # Replace padding token id with -100 so loss ignores padding
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         labels,
        }


# ─── MAIN TRAINING ────────────────────────────────────────────────────────────

def train():
    # 1. Load data
    print("\n[1/5] Loading dataset...")
    data = load_jsonl(JSONL_FILES, max_samples=MAX_SAMPLES)
    if not data:
        print("[ERROR] No data loaded. Check file paths.")
        return

    # Split train/val (90/10)
    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data   = data[split:]
    print(f"[DATA] Train: {len(train_data)}, Val: {len(val_data)}")

    # 2. Load tokenizer and model
    print(f"\n[2/5] Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,  # CPU needs float32
    )
    print(f"[MODEL] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 3. Apply LoRA
    print("\n[3/5] Applying LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q", "v"],   # attention query and value
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Create datasets and dataloaders
    print("\n[4/5] Creating datasets...")
    train_dataset = LegalDataset(train_data, tokenizer, MAX_INPUT_LEN, MAX_TARGET_LEN)
    val_dataset   = LegalDataset(val_data,   tokenizer, MAX_INPUT_LEN, MAX_TARGET_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    # 5. Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 6. Training loop
    print(f"\n[5/5] Training for {EPOCHS} epochs...")
    print(f"      Steps per epoch: {len(train_loader)}")
    print(f"      Total optimizer steps: {total_steps}")
    print(f"      Output: {OUTPUT_DIR}")
    print("-" * 60)

    model.train()
    best_val_loss = float("inf")
    global_step   = 0
    start_time    = time.time()

    for epoch in range(EPOCHS):
        epoch_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels         = batch["labels"]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / GRAD_ACCUM
            loss.backward()
            epoch_loss += outputs.loss.item()

            # Gradient accumulation
            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Progress
                elapsed = time.time() - start_time
                avg_loss = epoch_loss / (step + 1)
                steps_done = epoch * len(train_loader) + step + 1
                steps_total = EPOCHS * len(train_loader)
                eta = (elapsed / steps_done) * (steps_total - steps_done)

                print(
                    f"  Epoch {epoch+1}/{EPOCHS} | "
                    f"Step {step+1}/{len(train_loader)} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"ETA: {eta/60:.1f}min"
                )

                # Save checkpoint
                if global_step % SAVE_EVERY == 0:
                    ckpt_path = os.path.join(OUTPUT_DIR, f"checkpoint-{global_step}")
                    model.save_pretrained(ckpt_path)
                    print(f"  [SAVE] Checkpoint saved: {ckpt_path}")

        # Validation
        print(f"\n  Running validation...")
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                val_loss += outputs.loss.item()
        val_loss /= len(val_loader)
        print(f"  Epoch {epoch+1} Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(os.path.join(OUTPUT_DIR, "best"))
            tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "best"))
            print(f"  [BEST] New best model saved! Val loss: {val_loss:.4f}")

        model.train()

    # Save final model
    final_path = os.path.join(OUTPUT_DIR, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\n[DONE] Training complete!")
    print(f"       Best val loss: {best_val_loss:.4f}")
    print(f"       Model saved to: {final_path}")
    print(f"       Total time: {(time.time()-start_time)/60:.1f} minutes")

    # Quick test
    print("\n[TEST] Generating a sample summary...")
    model.eval()
    test_prompt = "Summarize the following Indian legal case document.\n\nThis is an appeal from the judgment of the High Court of Madras. The appellant was convicted under Section 302 IPC for murder. The Sessions Court sentenced him to death. The High Court confirmed the sentence. The appellant contends that the evidence is circumstantial and insufficient."
    inputs = tokenizer(test_prompt, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=150,
            num_beams=2,
            early_stopping=True,
        )
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"  Input: {test_prompt[:100]}...")
    print(f"  Output: {summary}")


if __name__ == "__main__":
    train()
