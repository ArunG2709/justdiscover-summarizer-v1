"""
LoRA Fine-tuning for Flan-T5 on Indian Legal Documents
Sources:
  1. JSONL dataset from Downloads (court judgments)
  2. PDF, DOCX, and text files from training_docs/
Model: google/flan-t5-base
Hardware: CPU
"""

import json
import os
import re
import io
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import time

# ─── CONFIG ───────────────────────────────────────────────────────────────────
JSONL_FILES = [
    "/home/tharani/Downloads/indian_legal_dataset.jsonl",
    "/home/tharani/Downloads/indian_legal_dataset (1).jsonl",
]
TRAINING_DOCS_DIR = "/home/tharani/justact_bert/training_docs"
MODEL_NAME        = "google/flan-t5-base"
OUTPUT_DIR        = "/home/tharani/justact_bert/lora_model_v2"
MAX_INPUT_LEN     = 512
MAX_TARGET_LEN    = 200
BATCH_SIZE        = 2
GRAD_ACCUM        = 8
EPOCHS            = 3
LR                = 3e-4
MAX_JSONL_SAMPLES = 500   # from JSONL
LORA_R            = 8
LORA_ALPHA        = 16
LORA_DROPOUT      = 0.05
SAVE_EVERY        = 50
SEED              = 42
# ──────────────────────────────────────────────────────────────────────────────

random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 60)
print("  LoRA Fine-tuning v2: JSONL + PDF/DOCX/OCR documents")
print("=" * 60)


# ─── TEXT EXTRACTION FROM FILES ───────────────────────────────────────────────

def extract_text_from_pdf(filepath):
    """Extract text from PDF using pdfplumber, fallback to OCR."""
    try:
        import pdfplumber
        with pdfplumber.open(filepath) as pdf:
            pages = []
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    pages.append(t)
            text = "\n".join(pages)
            if len(text.split()) > 50:
                return text
    except Exception as e:
        print(f"  [PDF] pdfplumber failed for {os.path.basename(filepath)}: {e}")

    # Fallback to OCR
    try:
        from pdf2image import convert_from_path
        import pytesseract
        print(f"  [OCR] Trying OCR for {os.path.basename(filepath)}...")
        images = convert_from_path(filepath, dpi=200)
        pages = []
        for img in images[:5]:   # limit to first 5 pages for speed
            t = pytesseract.image_to_string(img, lang='eng')
            if t.strip():
                pages.append(t)
        return "\n".join(pages)
    except Exception as e:
        print(f"  [OCR] Failed: {e}")
        return ""


def extract_text_from_docx(filepath):
    """Extract text from DOCX."""
    try:
        import docx
        doc = docx.Document(filepath)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    except Exception as e:
        print(f"  [DOCX] Failed for {os.path.basename(filepath)}: {e}")
        return ""


def extract_text_from_file(filepath):
    """Route to correct extractor based on file extension."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(filepath)
    elif ext in (".docx", ".doc"):
        return extract_text_from_docx(filepath)
    elif ext in (".txt", ".text"):
        try:
            with open(filepath, encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            print(f"  [TXT] Failed: {e}")
            return ""
    else:
        # Try as plain text for OCR output files
        try:
            with open(filepath, encoding="utf-8", errors="ignore") as f:
                return f.read()
        except:
            return ""


def clean_text(text):
    """Basic cleaning."""
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ─── SUMMARY GENERATION (rule-based for training labels) ──────────────────────

def detect_doc_type(text):
    """Simple doc type detection for labeling."""
    t = text.lower()
    if any(x in t for x in ["claimant", "respondent", "arbitration", "npa", "loan agreement"]):
        return "arbitration_claim"
    if any(x in t for x in ["the court held", "supreme court", "high court", "judgment"]):
        return "court_judgment"
    if any(x in t for x in ["income tax", "gst", "assessment year"]):
        return "tax_document"
    if any(x in t for x in ["section 138", "negotiable instruments", "cheque"]):
        return "cheque_bounce"
    return "general_legal"


def extract_claimant(text):
    """Extract claimant name from document."""
    patterns = [
        r'([A-Z][A-Za-z\s&.,Ltd]+?)\s*\.{2,}\s*(?:Claimant|Applicant|Petitioner|Plaintiff)',
        r'BETWEEN\s*:?\s*([A-Z][A-Za-z\s&.,Ltd]+?)\s+AND\s+',
        r'claimant\s+(?:is|named)\s+([A-Z][A-Za-z\s&.,Ltd]+?)[\.,]',
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            name = m.group(1).strip(' .,\n\t')
            name = re.sub(r'^(?:between|and)\s+', '', name, flags=re.IGNORECASE).strip()
            if len(name) > 3 and 'between' not in name.lower():
                return name
    return None


def extract_respondent(text):
    """Extract respondent name."""
    patterns = [
        r'([A-Z][A-Za-z\s&.,]+?)\s*\.{2,}\s*(?:Respondent|Defendant|Borrower)',
        r'AND\s+([A-Z][A-Za-z\s&.,]+?)\s*\.{2,}\s*(?:Respondent|Defendant)',
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            name = m.group(1).strip(' .,\n\t')
            if len(name) > 3:
                return name
    return None


def extract_amount(text):
    """Extract claim amount."""
    patterns = [
        r'Rs\.?\s*([\d,]+(?:\.\d{1,2})?)\s*ps?\s+(?:is\s+)?due',
        r'sum\s+of\s+Rs\.?\s*([\d,]+(?:\.\d{1,2})?)',
        r'NET\s+Amount[:\s]+Rs\.?\s*([\d,]+(?:\.\d{1,2})?)',
        r'amount\s+of\s+Rs\.?\s*([\d,]+(?:\.\d{1,2})?)',
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return f"Rs.{m.group(1)}"
    return None


def build_training_summary(text, doc_type):
    """
    Build a clean target summary for training.
    This becomes the 'output' label the model learns to generate.
    """
    words = text.split()
    word_count = len(words)

    if doc_type == "arbitration_claim":
        claimant   = extract_claimant(text) or "The claimant"
        respondent = extract_respondent(text) or "the respondent"
        amount     = extract_amount(text)

        # Build a natural summary
        summary = f"{claimant} has filed an arbitration claim against {respondent} "
        summary += "under the Arbitration and Conciliation Act, 1996. "

        if amount:
            summary += f"The total claim amount is {amount}. "

        # Add key facts from first 500 words
        sample = " ".join(words[:500])
        if "loan agreement" in sample.lower():
            summary += "The dispute arises out of a loan agreement between the parties. "
        if "default" in sample.lower() or "instalment" in sample.lower():
            summary += "The respondent has defaulted on payment of instalments. "
        if "hypothecation" in sample.lower():
            summary += "The vehicle was hypothecated as security for the loan. "
        if "venue" in sample.lower() or "chennai" in sample.lower():
            summary += "The venue of arbitration is Chennai. "

        summary += f"The document contains {word_count:,} words."
        return summary.strip()

    elif doc_type == "court_judgment":
        # Extract case details
        sample = " ".join(words[:300])
        summary = "This is a court judgment. "

        # Try to get parties
        vs_match = re.search(r'([A-Z][A-Za-z\s]+?)\s+vs?\s+([A-Z][A-Za-z\s]+?)[\.\n]', text)
        if vs_match:
            summary = f"This case involves {vs_match.group(1).strip()} versus {vs_match.group(2).strip()}. "

        if "appeal" in sample.lower():
            summary += "The matter is an appeal before a higher court. "
        if "conviction" in sample.lower() or "acquittal" in sample.lower():
            summary += "The case involves criminal proceedings. "
        if "section" in sample.lower():
            sec = re.search(r'[Ss]ection\s+(\d+[A-Z]?)', sample)
            if sec:
                summary += f"The case involves Section {sec.group(1)} of the relevant Act. "

        summary += f"The judgment contains {word_count:,} words."
        return summary.strip()

    elif doc_type == "cheque_bounce":
        claimant = extract_claimant(text) or "The complainant"
        amount   = extract_amount(text)
        summary  = f"{claimant} has filed a complaint for cheque dishonour "
        summary += "under Section 138 of the Negotiable Instruments Act. "
        if amount:
            summary += f"The cheque amount is {amount}. "
        return summary.strip()

    else:
        # General legal document
        summary = f"This is a legal document containing {word_count:,} words. "
        orgs = re.findall(r'[A-Z][A-Z\s&.,Ltd]{5,30}(?:Limited|Ltd|LLP|Pvt)', text)
        if orgs:
            summary += f"Parties involved include {orgs[0].strip()}. "
        return summary.strip()


# ─── LOAD ALL TRAINING DATA ───────────────────────────────────────────────────

def load_jsonl_data(paths, max_samples):
    """Load from JSONL files."""
    data = []
    for path in paths:
        if not os.path.exists(path):
            print(f"[WARN] JSONL not found: {path}")
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if obj.get("input") and obj.get("output"):
                        data.append({
                            "input":  obj["input"],
                            "output": obj["output"],
                            "source": "jsonl"
                        })
                except:
                    continue
        print(f"[DATA] JSONL loaded: {len(data)} total")

    random.shuffle(data)
    if len(data) > max_samples:
        data = data[:max_samples]
    return data


def load_training_docs(docs_dir):
    """Load and process PDF/DOCX/text files from training_docs."""
    data = []
    if not os.path.exists(docs_dir):
        print(f"[WARN] training_docs not found: {docs_dir}")
        return data

    files = [f for f in os.listdir(docs_dir)
             if os.path.isfile(os.path.join(docs_dir, f))]

    print(f"[DATA] Found {len(files)} files in training_docs/")

    for fname in files:
        fpath = os.path.join(docs_dir, fname)
        ext   = os.path.splitext(fname)[1].lower()

        if ext not in ('.pdf', '.docx', '.doc', '.txt', '.text', ''):
            continue

        print(f"  Processing: {fname}...")
        text = extract_text_from_file(fpath)
        text = clean_text(text)

        if len(text.split()) < 100:
            print(f"  → Too short, skipping")
            continue

        doc_type = detect_doc_type(text)
        summary  = build_training_summary(text, doc_type)

        # Build prompt
        prompt = f"Summarize the following Indian legal document.\n\n{' '.join(text.split()[:400])}"

        data.append({
            "input":  prompt,
            "output": summary,
            "source": f"training_docs/{fname}",
            "type":   doc_type,
        })
        print(f"  → {doc_type}: {len(text.split())} words → summary: {len(summary.split())} words")

    print(f"[DATA] Extracted {len(data)} training examples from training_docs/")
    return data


# ─── DATASET CLASS ────────────────────────────────────────────────────────────

class LegalDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data      = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item   = self.data[idx]
        prompt = item["input"]
        target = item["output"]

        # Truncate
        prompt_words = prompt.split()
        if len(prompt_words) > 400:
            prompt = " ".join(prompt_words[:400]) + "..."

        target_words = target.split()
        if len(target_words) > 150:
            target = " ".join(target_words[:150])

        enc = self.tokenizer(
            prompt,
            max_length=MAX_INPUT_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        dec = self.tokenizer(
            text_target=target,
            max_length=MAX_TARGET_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = dec["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         labels,
        }


# ─── MAIN TRAINING ────────────────────────────────────────────────────────────

def train():
    # 1. Load all data
    print("\n[1/5] Loading all training data...")

    jsonl_data  = load_jsonl_data(JSONL_FILES, MAX_JSONL_SAMPLES)
    docs_data   = load_training_docs(TRAINING_DOCS_DIR)
    all_data    = jsonl_data + docs_data

    print(f"\n[DATA] Total: {len(all_data)} examples")
    print(f"       JSONL: {len(jsonl_data)}")
    print(f"       Docs:  {len(docs_data)}")

    if not all_data:
        print("[ERROR] No data loaded!")
        return

    # Count by type
    types = {}
    for d in docs_data:
        t = d.get("type", "unknown")
        types[t] = types.get(t, 0) + 1
    print(f"       Doc types: {types}")

    # Shuffle and split
    random.shuffle(all_data)
    split      = int(0.9 * len(all_data))
    train_data = all_data[:split]
    val_data   = all_data[split:]
    print(f"       Train: {len(train_data)}, Val: {len(val_data)}")

    # 2. Load model
    print(f"\n[2/5] Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
    )

    # 3. Apply LoRA
    print("\n[3/5] Applying LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q", "v"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Datasets
    print("\n[4/5] Creating datasets...")
    train_dataset = LegalDataset(train_data, tokenizer)
    val_dataset   = LegalDataset(val_data,   tokenizer)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    # 5. Train
    optimizer   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n[5/5] Training for {EPOCHS} epochs...")
    print(f"      Steps per epoch: {len(train_loader)}")
    print(f"      Total steps: {total_steps}")
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
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss / GRAD_ACCUM
            loss.backward()
            epoch_loss += outputs.loss.item()

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                elapsed     = time.time() - start_time
                avg_loss    = epoch_loss / (step + 1)
                steps_done  = epoch * len(train_loader) + step + 1
                steps_total = EPOCHS * len(train_loader)
                eta         = (elapsed / steps_done) * (steps_total - steps_done)

                print(
                    f"  Epoch {epoch+1}/{EPOCHS} | "
                    f"Step {step+1}/{len(train_loader)} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"ETA: {eta/60:.1f}min"
                )

                if global_step % SAVE_EVERY == 0:
                    ckpt = os.path.join(OUTPUT_DIR, f"checkpoint-{global_step}")
                    model.save_pretrained(ckpt)
                    print(f"  [SAVE] {ckpt}")

        # Validation
        print(f"\n  Running validation...")
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                val_loss += out.loss.item()
        val_loss /= len(val_loader)
        print(f"  Epoch {epoch+1} Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(OUTPUT_DIR, "best")
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)
            print(f"  [BEST] Saved! Val loss: {val_loss:.4f}")

        model.train()

    # Save final
    final_path = os.path.join(OUTPUT_DIR, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    print(f"\n[DONE] Training complete!")
    print(f"       Best val loss: {best_val_loss:.4f}")
    print(f"       Saved to: {OUTPUT_DIR}")
    print(f"       Time: {(time.time()-start_time)/60:.1f} minutes")

    # Quick test
    print("\n[TEST] Sample generation...")
    model.eval()
    test = "Summarize the following Indian legal document.\n\nSundaram Finance Limited has filed an arbitration claim against M/S S S Famous Garden Furniture. The loan amount was Rs.834000. The respondent defaulted on instalments."
    inputs = tokenizer(test, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        out = model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=150,
            num_beams=2,
            early_stopping=True,
        )
    print(f"  Output: {tokenizer.decode(out[0], skip_special_tokens=True)}")


if __name__ == "__main__":
    train()
