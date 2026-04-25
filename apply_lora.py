"""
Run this AFTER training is complete.
Updates main.py to load the LoRA fine-tuned model instead of base Flan-T5.
"""

LORA_MODEL_PATH = "/home/tharani/justact_bert/lora_model/best"

patch = f'''
# ── Flan-T5 + LoRA (fine-tuned on Indian legal documents) ──────────────────
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from peft import PeftModel
    import torch

    T5_MODEL_NAME = "google/flan-t5-base"
    LORA_PATH     = "{LORA_MODEL_PATH}"

    T5_TOKENIZER = AutoTokenizer.from_pretrained(LORA_PATH)
    base_model   = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL_NAME, torch_dtype=torch.float32)
    T5_MODEL     = PeftModel.from_pretrained(base_model, LORA_PATH)
    T5_MODEL.eval()
    T5_LOADED = True
    print("[STARTUP] Flan-T5 + LoRA loaded ✓")
except Exception as e:
    print(f"[STARTUP] LoRA model failed, falling back to base T5: {{e}}")
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
        T5_MODEL_NAME = "google/flan-t5-base"
        T5_TOKENIZER  = AutoTokenizer.from_pretrained(T5_MODEL_NAME)
        T5_MODEL      = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL_NAME)
        T5_MODEL.eval()
        T5_LOADED = True
        print("[STARTUP] Flan-T5-base (base, no LoRA) loaded ✓")
    except Exception as e2:
        print(f"[STARTUP] Flan-T5 failed: {{e2}}")
'''

# Read main.py
with open("/home/tharani/justact_bert/main.py") as f:
    code = f.read()

# Replace the T5 loading block
old_block = '''# ── Flan-T5-base (summarization) ──────────────────────────────────────────
# T5 is a text-to-text model — reads document, generates natural summary
# Much better than rule-based templates for complex documents
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch

    T5_MODEL_NAME = "google/flan-t5-base"
    T5_TOKENIZER  = AutoTokenizer.from_pretrained(T5_MODEL_NAME)
    T5_MODEL      = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL_NAME)
    T5_MODEL.eval()
    T5_LOADED = True
    print("[STARTUP] Flan-T5-base loaded ✓")
except Exception as e:
    print(f"[STARTUP] Flan-T5 failed: {e}")'''

if old_block in code:
    code = code.replace(old_block, patch)
    with open("/home/tharani/justact_bert/main.py", "w") as f:
        f.write(code)
    print("✓ main.py updated to use LoRA model")
    print(f"  LoRA path: {LORA_MODEL_PATH}")
else:
    print("Could not find T5 loading block in main.py")
    print("Please manually replace the T5 loading section with:")
    print(patch)
