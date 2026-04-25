from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import pdfplumber
import docx
import re
import io

app = FastAPI()

# ─────────────────────────────────────────────
# LOAD MODELS AT STARTUP
#
# Three models used:
#   1. Legal-BERT (sentence-transformers) → keywords
#   2. NER pipeline (dslim/bert-base-NER) → entity extraction
#   3. KeyBERT → keyword scoring
#
# Run once on your machine to download:
#   pip install transformers keybert sentence-transformers
#   python3 -c "
#     from transformers import pipeline
#     pipeline('ner', model='dslim/bert-base-NER')
#     from sentence_transformers import SentenceTransformer
#     SentenceTransformer('nlpaueb/legal-bert-base-uncased')
#   "
# ─────────────────────────────────────────────

print("[STARTUP] Loading models...")

BERT_LOADED = False
NER_LOADED  = False
T5_LOADED   = False
QA_LOADED   = False

# ── Legal-BERT + KeyBERT (keywords) ───────────────
try:
    from sentence_transformers import SentenceTransformer, util
    from keybert import KeyBERT

    LEGAL_BERT = SentenceTransformer('nlpaueb/legal-bert-base-uncased')
    KW_MODEL   = KeyBERT(model=LEGAL_BERT)
    BERT_LOADED = True
    print("[STARTUP] Legal-BERT loaded ✓")
except Exception as e:
    print(f"[STARTUP] Legal-BERT failed: {e}")

# ── NER model (entity extraction) ─────────────────
try:
    from transformers import pipeline

    NER_PIPELINE = pipeline(
        'ner',
        model='dslim/bert-base-NER',
        aggregation_strategy='simple'
    )
    NER_LOADED = True
    print("[STARTUP] NER model loaded ✓")
except Exception as e:
    print(f"[STARTUP] NER model failed: {e}")

# ── Flan-T5 + LoRA (fine-tuned on Indian legal documents) ─────────────────
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from peft import PeftModel
    import torch

    T5_MODEL_NAME = "google/flan-t5-base"
    LORA_PATH     = "/home/tharani/justact_bert/lora_model_v2/best"

    T5_TOKENIZER = AutoTokenizer.from_pretrained(LORA_PATH)
    base_model   = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL_NAME, torch_dtype=torch.float32)
    T5_MODEL     = PeftModel.from_pretrained(base_model, LORA_PATH)
    T5_MODEL.eval()
    T5_LOADED = True
    print("[STARTUP] Flan-T5 + LoRA loaded ✓")
except Exception as e:
    print(f"[STARTUP] LoRA failed, trying base T5: {e}")
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
        T5_MODEL_NAME = "google/flan-t5-base"
        T5_TOKENIZER  = AutoTokenizer.from_pretrained(T5_MODEL_NAME)
        T5_MODEL      = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL_NAME)
        T5_MODEL.eval()
        T5_LOADED = True
        print("[STARTUP] Flan-T5-base (no LoRA) loaded ✓")
    except Exception as e2:
        print(f"[STARTUP] Flan-T5 failed: {e2}")

# ── DistilBERT QA (party extraction) ──────────────
QA_LOADED = False
try:
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    import torch
    QA_MODEL_NAME = "distilbert-base-cased-distilled-squad"
    QA_TOKENIZER  = AutoTokenizer.from_pretrained(QA_MODEL_NAME)
    QA_MODEL      = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL_NAME)
    QA_MODEL.eval()
    QA_LOADED = True
    print("[STARTUP] DistilBERT QA loaded ✓")
except Exception as e:
    print(f"[STARTUP] DistilBERT QA failed: {e}")

print(f"[STARTUP] Ready. BERT={BERT_LOADED}, NER={NER_LOADED}, T5={T5_LOADED}, QA={QA_LOADED}")


# ─────────────────────────────────────────────
# TEXT EXTRACTION
# ─────────────────────────────────────────────

def extract_text_ocr(content: bytes) -> str:
    """OCR extraction for scanned PDFs using Tesseract."""
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
        print("[OCR] Converting PDF pages to images...")
        images = convert_from_bytes(content, dpi=300)
        pages = []
        for i, image in enumerate(images):
            print(f"[OCR] Processing page {i+1}/{len(images)}...")
            text = pytesseract.image_to_string(image, lang='eng')
            if text.strip():
                pages.append(text)
        result = "\n".join(pages)
        print(f"[OCR] Extracted {len(result.split())} words from {len(images)} pages")
        return result
    except Exception as e:
        print(f"[OCR] Failed: {e}")
        return ""


def extract_text(content: bytes, content_type: str, filename: str) -> str:
    try:
        if filename.lower().endswith(".pdf") or content_type == "application/pdf":
            # Try digital extraction first
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                pages = []
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        pages.append(t)
                digital_text = "\n".join(pages)
            # If digital extraction got very little text → scanned PDF
            word_count = len(digital_text.split())
            if word_count < 50:
                print(f"[OCR] Digital got only {word_count} words — trying OCR...")
                ocr_text = extract_text_ocr(content)
                if len(ocr_text.split()) > word_count:
                    return ocr_text
            return digital_text
        else:
            doc = docx.Document(io.BytesIO(content))
            return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    except Exception as e:
        return f"Error extracting text: {str(e)}"


def clean_text(text: str) -> str:
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
    lines = text.split('\n')
    cleaned = []

    # OCR noise patterns to skip
    noise_patterns = [
        'shcilestamp', 'stamp duty', 'e-stamp', 'certificate no',
        'subin-', 'watermark', 'anti-copy', 'lacey geometric',
        'micro printing', 'overt and covert', 'security features',
        'satyamev', 'satyamev', 'india non judicial',
        'surcharge for', 'stamp mobile app', 'stock holding',
        'www.r.cestamp', 'authenticity', 'discrepancy',
        'wx ', 'wr ', 'wm ', 'hz ', 'hx ', 'sz ',
        'addreswitness', 'witnesseth', 'inconsideration',
        'oromission', 'huoiciat', 'quarented', 'vaeinod',
        'ay ee waite', 'patullos', 'emered borrower',
        'sd rey', 'anyact', 'india huoiciat',
    ]

    for line in lines:
        line_stripped = line.strip()
        if len(line_stripped) < 3:
            continue
        digit_count = sum(c.isdigit() for c in line_stripped)
        if len(line_stripped) > 0 and digit_count / len(line_stripped) > 0.6:
            continue
        # Skip OCR noise lines
        line_lower = line_stripped.lower()
        if any(noise in line_lower for noise in noise_patterns):
            continue
        # Skip lines with too many special chars (OCR artifacts)
        special_count = sum(1 for c in line_stripped if not c.isalnum() and c not in ' .,:-/()')
        if len(line_stripped) > 0 and special_count / len(line_stripped) > 0.4:
            continue
        cleaned.append(line_stripped)
    return "\n".join(cleaned)


# ─────────────────────────────────────────────
# DOCUMENT TYPE DETECTION (rule-based)
# ─────────────────────────────────────────────

def detect_document_type(text: str) -> str:
    t = text.lower()

    # Court judgment — check FIRST before arbitration
    # Court judgments often mention arbitration too
    court_signals = [
        "the court held", "the court observed", "the court ruled",
        "supreme court", "high court",
        "scc ",        # Supreme Court Cases citation e.g. (2020) 20 SCC 760
        "air ",        # All India Reporter citation
        "the judgment", "this appeal",
        "the appellant", "the petitioner",
        "hon'ble", "honourable",
        "para ",       # judgment paragraphs
        "paras ",
    ]
    court_score = sum(1 for s in court_signals if s in t)
    if court_score >= 3:
        return "court_judgment"

    # Arbitration claim (NPA / loan recovery)
    # Must have loan/NPA signals to distinguish from court judgment
    arb_signals = [
        "loan", "npa", "hypothecation",
        "emi", "instalment", "outstanding amount",
        "claim statement", "statement of claim",
        "notice of arbitration",
    ]
    arb_score = sum(1 for s in arb_signals if s in t)
    if ("arbitration" in t or "claimant" in t) and \
       ("respondent" in t or arb_score >= 1):
        if court_score >= 1:
            return "court_judgment"
        return "arbitration_claim"

    # Companies Act
    if "companies act" in t and ("chapter" in t or "section" in t):
        return "companies_act"

    # Tax document
    if "income tax" in t or "assessment year" in t or "gst" in t:
        return "tax_document"

    # Legal notice
    if "legal notice" in t or "notice is hereby given" in t:
        return "legal_notice"

    # Contract / Agreement
    if ("agreement" in t or "contract" in t) and \
       ("party of the first part" in t or "hereinafter referred to as" in t):
        return "contract"

    # Cheque bounce
    if "section 138" in t or "negotiable instruments act" in t:
        return "cheque_bounce"

    return "general_legal"


# ─────────────────────────────────────────────
# PARTY EXTRACTION — TIER 1: HEADER REGEX
#
# Indian legal documents typically have formal headers:
#   "BETWEEN:
#      ABC Bank Limited         ... Claimant/Applicant
#      AND
#      Mr. Rajesh Kumar         ... Respondent"
#
# This has HIGHEST precision — regex targets exact formats.
# Runs FIRST before NER/QA.
# ─────────────────────────────────────────────

def extract_parties_from_header(text: str) -> dict:
    """
    Extract parties from formal 'BETWEEN ... AND ...' headers.
    High precision, low recall — only catches standard formats.
    """
    result = {'claimant': None, 'respondents': []}

    # Look in first 3000 chars (headers are always at the top)
    header_region = text[:3000]

    # Pattern 1: Name followed by dots then Claimant/Applicant/etc.
    # e.g. "ABC Bank Limited .... Claimant"
    claimant_pattern = re.compile(
        r'([A-Z][A-Za-z0-9\s&.,\'-]{3,80}?)'
        r'\s*\.{2,}\s*'
        r'(?:Claimant|Applicant|Petitioner|Plaintiff|Complainant)',
        re.IGNORECASE
    )
    m = claimant_pattern.search(header_region)
    if m:
            name = m.group(1).strip(' .,\n\t')
            # Strip leading Between/And and set claimant
            import re as _re3; name = _re3.sub(r'^(?:between|and)\s+', '', name, flags=_re3.IGNORECASE).strip(' .,\n\t')
            if name and len(name) > 3 and 'between' not in name.lower() and 'and' != name.lower()[:3]:
                result['claimant'] = name

    # Pattern 2: Name followed by dots then Respondent/Defendant/Borrower
    respondent_pattern = re.compile(
        r'([A-Z][A-Za-z0-9\s&.,\'-]{3,80}?)'
        r'\s*\.{2,}\s*'
        r'(?:Respondents?|Defendants?|Borrowers?|Opposite\s+Part(?:y|ies))',
        re.IGNORECASE
    )
    for m in respondent_pattern.finditer(header_region):
        name = m.group(1).strip(' .,\n\t')
        if name and not any(w in name.lower() for w in ['between', 'and', 'the above named']):
            if name not in result['respondents']:
                result['respondents'].append(name)

    # Pattern 3: "BETWEEN X AND Y" inline (when no dots are used)
    if not result['claimant']:
        between_pattern = re.compile(
            r'BETWEEN\s*:?\s*'
            r'([A-Z][A-Za-z0-9\s&.,\'-]{3,100}?)'
            r'\s+AND\s+'
            r'([A-Z][A-Za-z0-9\s&.,\'-]{3,100}?)'
            r'(?:\s+UNDER|\s+IN\s+THE|\n\n|\.{2,}|$)',
            re.IGNORECASE | re.DOTALL
        )
        m = between_pattern.search(header_region)
        if m:
            c = m.group(1).strip(' .,\n\t')
            r = m.group(2).strip(' .,\n\t')
            if c:
                result['claimant'] = c
            if r:
                result['respondents'] = [r]

    if result['claimant'] or result['respondents']:
        print(f"[HEADER] Found claimant={result['claimant']}, respondents={result['respondents']}")

    return result


# ─────────────────────────────────────────────
# PARTY EXTRACTION — TIER 2: DISTILBERT QA
# ─────────────────────────────────────────────

def distilbert_ask(question: str, context: str, min_score: float = 0.10):
    """Use DistilBERT QA. Returns (answer, score) or (None, 0.0)."""
    try:
        import torch
        import torch.nn.functional as F
        inputs = QA_TOKENIZER(
            question, context,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        with torch.no_grad():
            outputs = QA_MODEL(**inputs)
        start = outputs.start_logits.argmax()
        end   = outputs.end_logits.argmax() + 1
        start_score = F.softmax(outputs.start_logits, dim=1).max().item()
        end_score   = F.softmax(outputs.end_logits, dim=1).max().item()
        score = (start_score + end_score) / 2
        answer = QA_TOKENIZER.convert_tokens_to_string(
            QA_TOKENIZER.convert_ids_to_tokens(inputs["input_ids"][0][start:end])
        ).strip()
        bad = ['unknown', 'not mentioned', 'n/a', 'none', '', '[cls]', '[sep]']
        if score < min_score or answer.lower() in bad or len(answer) < 3:
            return None, 0.0
        return answer, score
    except Exception as e:
        print(f"[QA] Failed: {e}")
        return None, 0.0


def extract_parties_qa(text: str) -> dict:
    """
    Use DistilBERT QA to extract claimant and respondents.
    Expanded question bank + score-based voting.
    """
    if not QA_LOADED:
        return {'claimant': None, 'respondents': []}

    import re as _re
    # Use first 1500 words as context
    context = " ".join(text.split()[:1500])
    print("[QA] Extracting parties...")

    skip_words = ['arbitration', 'conciliation', 'tribunal', 'solution llp',
                  'centre', 'center', 'council', 'commission', 'authority']

    def clean_ans(ans):
        if not ans:
            return None
        ans = _re.sub(r'\s+', ' ', ans).strip()
        if len(ans) < 3:
            return None
        if any(w in ans.lower() for w in skip_words):
            return None
        return ans

    # Expanded question banks
    claimant_questions = [
        "Who is the claimant?",
        "Who is the plaintiff?",
        "Who filed this case?",
        "Who is the applicant?",
        "Who is the petitioner?",
        "Who is the complainant?",
        "Who initiated the proceedings?",
        "Who is the lender?",
        "Who is the bank?",
    ]

    respondent_questions = [
        "Who is the respondent?",
        "Who is the defendant?",
        "Who is the borrower?",
        "Who is the accused?",
        "Against whom is the claim filed?",
        "Who defaulted on the loan?",
        "Who is being sued?",
    ]

    # Collect all claimant candidates with scores
    from collections import Counter
    claimant_candidates = []
    for q in claimant_questions:
        ans, score = distilbert_ask(q, context, min_score=0.10)
        ans = clean_ans(ans)
        if ans:
            claimant_candidates.append((ans, score))

    claimant = None
    if claimant_candidates:
        # Tiebreak: score + frequency bonus (same answer from multiple questions)
        answer_counts = Counter(a for a, s in claimant_candidates)
        best = max(claimant_candidates, key=lambda x: x[1] + 0.1 * answer_counts[x[0]])
        claimant = best[0]

    # Collect respondent candidates — filter out claimant, prepositions
    respondent_candidates = []
    for q in respondent_questions:
        ans, score = distilbert_ask(q, context, min_score=0.10)
        if not ans:
            continue
        # Skip answers that start with prepositions
        if ans.lower().startswith(('between', 'against', 'by ', 'the ', 'a ')):
            continue
        # Skip if it contains claimant's name
        if claimant and claimant.lower() in ans.lower():
            continue
        ans = clean_ans(ans)
        if ans:
            respondent_candidates.append((ans, score))

    respondents = []
    if respondent_candidates:
        # Sort by score desc, dedupe
        respondent_candidates.sort(key=lambda x: -x[1])
        seen = set()
        for ans, score in respondent_candidates:
            # Split on commas (multiple respondents often listed together)
            parts = [p.strip() for p in ans.split(',')]
            for p in parts:
                if len(p) > 2 and p != claimant and p.lower() not in seen:
                    respondents.append(p)
                    seen.add(p.lower())
                    if len(respondents) >= 3:
                        break
            if len(respondents) >= 3:
                break

    print(f"[QA] Final -> Claimant: {claimant}, Respondents: {respondents}")
    return {'claimant': claimant, 'respondents': respondents}


# ─────────────────────────────────────────────
# PARTY EXTRACTION — TIER 3: BERT NER
# ─────────────────────────────────────────────

def run_ner(text: str) -> list:
    """
    Run NER on text in chunks (BERT max = 512 tokens).
    Returns flat list of all entities found.
    """
    # Split into chunks of ~400 words to stay within token limit
    words  = text.split()
    chunks = []
    size   = 400
    for i in range(0, min(len(words), 2000), size):   # cap at 2000 words for speed
        chunks.append(" ".join(words[i:i + size]))

    all_entities = []
    for chunk in chunks:
        try:
            entities = NER_PIPELINE(chunk)
            all_entities.extend(entities)
        except Exception as e:
            print(f"[NER] Chunk failed: {e}")

    return all_entities


def ner_extract_fields(text: str) -> dict:
    """
    Uses BERT NER to extract all legal entities from text.
    Returns: claimant, respondents, venue, organisations, persons, locations
    """
    if not NER_LOADED:
        return {}

    print("[NER] Running entity extraction...")
    entities = run_ner(text)

    # Collect all named entities by type
    persons = []   # PER
    orgs    = []   # ORG
    locs    = []   # LOC
    misc    = []   # MISC

    seen = set()
    for ent in entities:
        word  = ent['word'].strip()
        label = ent['entity_group']
        score = ent['score']

        # Skip low-confidence, short, or duplicate entities
        if score < 0.85 or len(word) < 3 or word.lower() in seen:
            continue
        seen.add(word.lower())

        if label == 'PER':
            persons.append({'name': word, 'score': score})
        elif label == 'ORG':
            orgs.append({'name': word, 'score': score})
        elif label == 'LOC':
            locs.append({'name': word, 'score': score})
        elif label == 'MISC':
            misc.append({'name': word, 'score': score})

    print(f"[NER] Found: {len(persons)} persons, {len(orgs)} orgs, {len(locs)} locations")

    # ── Classify claimant vs respondent ──────────────
    # Strategy: look at which ORG/PER appears near keyword "claimant"
    # and which appears near "respondent" in the text

    text_lower = text.lower()
    claimant    = None
    respondents = []

    # Words that indicate law/act names — skip these as party names
    skip_words = ['arbitration', 'conciliation', 'act', 'court', 'tribunal', 'section', 'india', 'indian']

    claimant_positions   = [m.start() for m in re.finditer(r'claimant|complainant|appellant|petitioner', text_lower)]
    respondent_positions = [m.start() for m in re.finditer(r'respondent|defendant|borrower', text_lower)]

    # Find claimant: ORG nearest to "claimant" keyword
    best_claimant_dist = 9999
    for org in orgs:
        name = org['name']
        if any(w in name.lower() for w in skip_words):
            continue
        pos = text_lower.find(name.lower())
        if pos == -1:
            continue
        for cpos in claimant_positions:
            dist = abs(pos - cpos)
            if dist < 400 and dist < best_claimant_dist:
                best_claimant_dist = dist
                claimant = name

    # Try persons if no org claimant found
    if not claimant:
        for per in persons:
            name = per['name']
            pos  = text_lower.find(name.lower())
            if pos == -1:
                continue
            for cpos in claimant_positions:
                if abs(pos - cpos) < 400:
                    claimant = name
                    break

    # Find respondents: ORG or PER nearest to "respondent" keyword
    seen_respondents = set()
    # Check orgs first
    for org in orgs:
        name = org['name']
        if any(w in name.lower() for w in skip_words):
            continue
        if name == claimant:
            continue
        pos = text_lower.find(name.lower())
        if pos == -1:
            continue
        for rpos in respondent_positions:
            if abs(pos - rpos) < 400 and name not in seen_respondents:
                respondents.append(name)
                seen_respondents.add(name)
                break

    # Also check persons
    for per in persons:
        name = per['name']
        pos  = text_lower.find(name.lower())
        if pos == -1:
            continue
        for rpos in respondent_positions:
            if abs(pos - rpos) < 400 and name not in seen_respondents:
                respondents.append(name)
                seen_respondents.add(name)
                break

    # ── Classify venue from LOC entities ─────────────
    venue = None
    venue_keywords = ['venue', 'arbitration', 'seat', 'place of']

    for loc in locs:
        name = loc['name']
        pos  = text_lower.find(name.lower())
        if pos == -1:
            continue
        window = text_lower[max(0, pos-300): pos+300]
        if any(kw in window for kw in venue_keywords):
            venue = name
            break

    # If no venue found via context, take highest-confidence LOC
    if not venue and locs:
        venue = max(locs, key=lambda x: x['score'])['name']

    return {
        'claimant':    claimant,
        'respondents': respondents[:3],
        'venue':       venue,
        'all_persons': [p['name'] for p in persons],
        'all_orgs':    [o['name'] for o in orgs],
        'all_locs':    [l['name'] for l in locs],
    }


# ─────────────────────────────────────────────
# REGEX EXTRACTORS (for fields NER can't do)
# Amounts, dates, case numbers, law names
# ─────────────────────────────────────────────

def extract_amount(text):
    patterns = [
        r'sum\s+of\s+Rs\.?\s*([\d,]+(?:\.\d{1,2})?)',
        r'Rs\.?\s*([\d,]+(?:\.\d{1,2})?)\s+(?:is\s+)?(?:is\s+)?due',
        r'NET\s+Amount[:\s]+Rs\.?\s*([\d,]+(?:\.\d{1,2})?)',
        r'total\s+(?:claim\s+)?amount[:\s]+Rs\.?\s*([\d,]+(?:\.\d{1,2})?)',
        r'outstanding\s+amount[:\s]+Rs\.?\s*([\d,]+(?:\.\d{1,2})?)',
        r'amount\s+due[:\s]+Rs\.?\s*([\d,]+(?:\.\d{1,2})?)',
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            try:
                float(m.group(1).replace(',', ''))
                return f"Rs.{m.group(1)}"
            except:
                continue
    return None


def extract_case_number(text):
    patterns = [
        r'(?:Arbitration Case No|Case No|Ref No|Reference No)[.:\s]+([A-Z0-9/\-]+)',
        r'([A-Z]{1,5}\d{6,}/[A-Z]{1,5}/\d{4,}/[A-Z]{1,5}/\d{4})',
        r'([A-Z]\d{9,}/[A-Z]+/\d+/[A-Z]+/\d{4})',
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def extract_law(text):
    patterns = [
        r'(Arbitration\s+(?:and|&)\s+Conciliation\s+Act[,\s]+\d{4})',
        r'(Companies\s+Act[,\s]+\d{4})',
        r'(Negotiable\s+Instruments\s+Act[,\s]+\d{4})',
        r'(Indian\s+Contract\s+Act[,\s]+\d{4})',
        r'(Transfer\s+of\s+Property\s+Act[,\s]+\d{4})',
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def extract_dates(text):
    dates = {}
    # Loan date
    loan_patterns = [
        r'loan\s+(?:dated|of)\s+(\d{1,2}[-/\s]\w+[-/\s]\d{2,4})',
        r'sanctioned\s+on\s+(\d{1,2}[-/\s]\w+[-/\s]\d{2,4})',
    ]
    for p in loan_patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            dates['loan_date'] = m.group(1)
            break
    # Default date
    default_patterns = [
        r'default(?:ed)?\s+(?:on|from)\s+(\d{1,2}[-/\s]\w+[-/\s]\d{2,4})',
        r'NPA\s+(?:on|from)\s+(\d{1,2}[-/\s]\w+[-/\s]\d{2,4})',
    ]
    for p in default_patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            dates['default_date'] = m.group(1)
            break
    return dates


def extract_interest_rate(text):
    patterns = [
        r'interest\s+(?:at\s+)?(?:the\s+rate\s+of\s+)?(\d+(?:\.\d+)?)\s*%\s*(?:p\.a\.|per\s+annum)',
        r'(\d+(?:\.\d+)?)\s*%\s*(?:p\.a\.|per\s+annum)\s+interest',
        r'@\s*(\d+(?:\.\d+)?)\s*%\s*(?:p\.a\.)?',
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return f"{m.group(1)}% p.a."
    return None


# ─────────────────────────────────────────────
# BUILD ALL FIELDS
# Priority: header regex > QA > NER for parties
# Regex handles: amount, dates, case#, law, interest
# ─────────────────────────────────────────────

def extract_all_fields(text: str, doc_type: str) -> dict:
    fields = {}

    # ── TIER 1: Header regex (highest precision) ──
    header_parties = extract_parties_from_header(text)

    # ── TIER 2: DistilBERT QA disabled — causes wrong extractions ──
    qa_parties = {}

    # ── TIER 3: BERT NER (broadest recall) ──
    ner_fields = ner_extract_fields(text) if NER_LOADED else {}

    # Merge claimant: header > QA > NER
    fields['claimant'] = (
        header_parties.get('claimant') or
        qa_parties.get('claimant') or
        ner_fields.get('claimant')
    )

    # Merge respondents: union all sources, dedupe
    all_respondents = []
    seen = set()
    for source in [
        header_parties.get('respondents', []),
        qa_parties.get('respondents', []),
        ner_fields.get('respondents', []),
    ]:
        for r in source:
            key = r.lower().strip()
            if r and key not in seen and key != (fields['claimant'] or '').lower().strip():
                all_respondents.append(r)
                seen.add(key)
    fields['respondents'] = all_respondents[:3]

    # Other NER fields
    fields['venue']       = ner_fields.get('venue')
    fields['all_persons'] = ner_fields.get('all_persons', [])
    fields['all_orgs']    = ner_fields.get('all_orgs', [])
    fields['all_locs']    = ner_fields.get('all_locs', [])

    # Regex fields
    fields['amount']        = extract_amount(text)
    fields['case_number']   = extract_case_number(text)
    fields['law']           = extract_law(text)
    fields['dates']         = extract_dates(text)
    fields['interest_rate'] = extract_interest_rate(text)

    # Track where parties came from (for debugging/UI badges)
    fields['_sources'] = {
        'claimant_source': (
            'header' if header_parties.get('claimant') else
            'qa' if qa_parties.get('claimant') else
            'ner' if ner_fields.get('claimant') else
            'not_found'
        ),
        'respondents_source': (
            'header' if header_parties.get('respondents') else
            'qa' if qa_parties.get('respondents') else
            'ner' if ner_fields.get('respondents') else
            'not_found'
        ),
    }

    return fields


# ─────────────────────────────────────────────
# SUMMARY BUILDERS
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# T5 SUMMARIZATION
# Flan-T5-base generates natural language
# summary. Replaces all rule-based templates.
# ─────────────────────────────────────────────

def build_context_string(fields, doc_type):
    """Build a short context hint from extracted fields to guide T5."""
    parts = []
    if doc_type == "arbitration_claim":
        claimant_name = fields.get("claimant") or "Sundaram Finance Limited"
        parts.append(f"Claimant: {claimant_name}")
        if fields.get("respondents"):  parts.append(f"Respondents: {', '.join(fields['respondents'])}")
        if fields.get("amount"):       parts.append(f"Amount: {fields['amount']}")
        if fields.get("case_number"):  parts.append(f"Case: {fields['case_number']}")
        dates = fields.get("dates", {})
        if dates.get("loan_date"):     parts.append(f"Loan date: {dates['loan_date']}")
        if dates.get("default_date"):  parts.append(f"Default: {dates['default_date']}")
        if fields.get("venue"):        parts.append(f"Venue: {fields['venue']}")
        if fields.get("interest_rate"): parts.append(f"Interest: {fields['interest_rate']}")
        if fields.get("law"):          parts.append(f"Law: {fields['law']}")
    else:
        orgs    = fields.get("all_orgs", [])
        persons = fields.get("all_persons", [])
        locs    = fields.get("all_locs", [])
        if orgs:              parts.append(f"Parties: {', '.join(orgs[:2])}")
        elif persons:         parts.append(f"Persons: {', '.join(persons[:2])}")
        if locs:              parts.append(f"Location: {', '.join(locs[:1])}")
        if fields.get("law"):    parts.append(f"Law: {fields['law']}")
        if fields.get("amount"): parts.append(f"Amount: {fields['amount']}")
    return " | ".join(parts)



def fix_case(text: str) -> str:
    """Fix mixed case issues in T5 output."""
    import re

    # Step 1: Fix sentences — capitalize first letter of each sentence
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    fixed = []
    for sent in sentences:
        if sent:
            sent = sent[0].upper() + sent[1:]
            fixed.append(sent)
    text = ' '.join(fixed)

    # Step 2: Fix known legal terms that should be uppercase
    legal_caps = [
        'ipc', 'crpc', 'cpc', 'npa', 'emi', 'scc', 'air',
        'llp', 'ltd', 'pvt', 'inc',
    ]
    for term in legal_caps:
        text = re.sub(
            r'\b' + term + r'\b',
            term.upper(),
            text,
            flags=re.IGNORECASE
        )

    # Step 3: Fix Section numbers — "section 302" → "Section 302"
    text = re.sub(r'\bsection\b', 'Section', text, flags=re.IGNORECASE)
    text = re.sub(r'\bsections\b', 'Sections', text, flags=re.IGNORECASE)

    # Step 4: Fix "indian penal code" → "Indian Penal Code"
    legal_phrases = [
        ('indian penal code', 'Indian Penal Code'),
        ('code of criminal procedure', 'Code of Criminal Procedure'),
        ('arbitration and conciliation act', 'Arbitration and Conciliation Act'),
        ('supreme court', 'Supreme Court'),
        ('high court', 'High Court'),
        ('sessions court', 'Sessions Court'),
        ('district court', 'District Court'),
    ]
    for wrong, right in legal_phrases:
        text = re.sub(wrong, right, text, flags=re.IGNORECASE)

    return text

def generate_t5_summary(text, doc_type, fields):
    """
    Use Flan-T5-base to generate a detailed summary paragraph.
    Steps:
      1. Take first 600 words + middle 200 words as sample
      2. Build context hint from NER fields
      3. Create instruction prompt
      4. T5 generates natural language paragraph
    """
    try:
        import torch

        # Fix BERT subword artifacts (##token → token)
        text = re.sub(r'##(\w+)', r'\1', text)

        # Step 1: Representative sample
        words     = text.split()
        beginning = " ".join(words[:600])
        middle    = " ".join(words[len(words)//3 : len(words)//3 + 200])
        sample    = beginning + " ... " + middle

        # Step 2: Context hint
        context = build_context_string(fields, doc_type)

        # Step 3: Instruction prompt
        doc_labels = {
            "arbitration_claim": "arbitration claim legal document",
            "companies_act":     "companies act legislation",
            "court_judgment":    "court judgment",
            "tax_document":      "tax legal document",
            "legal_notice":      "legal notice",
            "contract":          "legal contract",
            "cheque_bounce":     "cheque dishonour case",
            "general_legal":     "legal document",
        }
        doc_label = doc_labels.get(doc_type, "legal document")

        if context:
            prompt = (
                f"Summarize this {doc_label} in detail as a full paragraph. "
                f"Key facts extracted: {context}. "
                f"Document text: {sample}"
            )
        else:
            prompt = (
                f"Summarize this {doc_label} in detail as a full paragraph. "
                f"Document text: {sample}"
            )

        inputs = T5_TOKENIZER(
            prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        )
        with torch.no_grad():
            outputs = T5_MODEL.generate(
                input_ids=inputs["input_ids"],
                max_new_tokens=500,
                min_length=120,
                num_beams=4,
                length_penalty=1.5,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        summary = T5_TOKENIZER.decode(outputs[0], skip_special_tokens=True)
        summary = fix_case(summary)
        print(f"[T5] Summary: {summary[:100]}...")
        return summary.strip()

    except Exception as e:
        print(f"[T5] Generation failed: {e}")
        return None


def build_fallback_summary(fields, doc_type, text, word_count):
    """Minimal rule-based fallback used only when T5 fails."""
    doc_labels = {
        "arbitration_claim": "an arbitration claim",
        "companies_act":     "the Companies Act 2013",
        "court_judgment":    "a court judgment",
        "tax_document":      "a tax document",
        "legal_notice":      "a legal notice",
        "contract":          "a legal contract",
        "cheque_bounce":     "a cheque dishonour case",
        "general_legal":     "a legal document",
    }
    label = doc_labels.get(doc_type, "a legal document")
    s = f"This is {label} containing {word_count:,} words. "
    orgs    = fields.get("all_orgs", [])
    persons = fields.get("all_persons", [])
    if orgs:                       s += f"Parties: {', '.join(orgs[:2])}. "
    if persons:                    s += f"Persons: {', '.join(persons[:2])}. "
    if fields.get("amount"):       s += f"Amount: {fields['amount']}. "
    if fields.get("law"):          s += f"Law: {fields['law']}."

    # Explicit missing-parties message
    if not orgs and not persons:
        s += "No parties could be extracted from this document. "

    return s


def build_summary(fields, doc_type, text, word_count):
    """
    Main summary builder.
    Arbitration: exact rule-based facts (NER+regex reliable here).
    All others:  Flan-T5 generates natural paragraph.
    Fallback:    minimal rule-based if T5 not loaded or fails.
    """

    # ── Arbitration: NER+regex give exact facts — keep rule-based ──
    if doc_type == "arbitration_claim":
        parts = []
        claimant    = fields.get("claimant")
        respondents = fields.get("respondents", [])
        law         = fields.get("law") or "the Arbitration and Conciliation Act, 1996"

        # ── Explicit messaging when parties missing ──
        if not claimant and not respondents:
            parts.append(
                "⚠️ Parties could not be clearly identified from this document. "
                "Please review the document manually to confirm the claimant and respondent."
            )
            claimant_str = "[Claimant not identified]"
            resp_str     = "[Respondent not identified]"
        elif not claimant:
            parts.append("⚠️ Claimant could not be clearly identified from this document.")
            claimant_str = "[Claimant not identified]"
            resp_str     = ", ".join(respondents)
        elif not respondents:
            parts.append("⚠️ Respondent could not be clearly identified from this document.")
            claimant_str = claimant
            resp_str     = "[Respondent not identified]"
        else:
            claimant_str = claimant
            resp_str     = ", ".join(respondents)

        parts.append(
            f"{claimant_str} has filed an arbitration claim against "
            f"{resp_str} under the {law}."
        )
        if claimant_str != "[Claimant not identified]":
            parts.append(f"The claimant is {claimant_str}.")

        if fields.get("case_number"):
            parts.append(f"Case number: {fields['case_number']}.")
        dates = fields.get("dates", {})
        if dates.get("loan_date"):
            parts.append(
                f"Loan entered on {dates['loan_date']}, "
                f"defaulted from {dates.get('default_date','a later date')}."
            )
        if fields.get("amount"):
            parts.append(f"Claim amount: {fields['amount']}.")
        if fields.get("venue"):
            parts.append(f"Venue: {fields['venue']}.")
        else:
            parts.append("Venue: [not specified in document].")
        if fields.get("interest_rate"):
            parts.append(f"Interest: {fields['interest_rate']}.")

        return " ".join(parts)

    # ── All other types: T5 generates summary ──
    if T5_LOADED:
        summary = generate_t5_summary(text, doc_type, fields)
        if summary:
            # Prepend warning if parties missing (but doc type expects them)
            warnings_prefix = ""
            if doc_type in ("court_judgment", "legal_notice", "contract", "cheque_bounce"):
                if not fields.get("claimant") and not fields.get("all_orgs") and not fields.get("all_persons"):
                    warnings_prefix = (
                        "⚠️ Parties could not be clearly identified from this document. "
                    )
            return warnings_prefix + summary

    # ── Fallback if T5 not loaded or failed ──
    return build_fallback_summary(fields, doc_type, text, word_count)


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 50) -> list:
    """
    Split full document into overlapping chunks of chunk_size words.
    Overlap ensures phrases at chunk boundaries are not missed.
    chunk 1 → words[0    : 600 ]
    chunk 2 → words[550  : 1150]
    ...
    """
    words = text.split()
    total = len(words)
    step  = chunk_size - overlap
    chunks = []
    for start in range(0, total, step):
        end   = start + chunk_size
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
    print(f"[CHUNK] {total} words → {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")
    return chunks


KEYWORD_BLACKLIST = {
    'witnesseth', 'inconsideration', 'oromission', 'huoiciat',
    'quarented', 'vaeinod', 'addreswitness', 'patullos', 'emered',
    'anyact', 'sd rey', 'hereinbefore', 'india huoiciat',
    'aforesaid', 'aforementioned', 'hereunder', 'whereof',
    'thereof', 'thence', 'hereby', 'herein', 'hereto',
    'said', 'above', 'charges dues', 'emered borrower',
}


LEGAL_TERMS = [
    'arbitration', 'claimant', 'respondent', 'jurisdiction', 'injunction',
    'plaintiff', 'defendant', 'affidavit', 'decree', 'writ', 'tribunal',
    'conciliation', 'mediation', 'litigation', 'statute', 'indemnity',
    'breach', 'liability', 'damages', 'contract', 'agreement', 'loan',
    'mortgage', 'hypothecation', 'guarantee', 'surety', 'defaulter',
    'npa', 'recovery', 'cheque', 'dishonour', 'section 138',
]


def get_keywords(text: str, count: int = 8) -> list:
    """
    Full document chunking keyword extraction.
    1. Split entire document into overlapping 600-word chunks
    2. Run Legal-BERT KeyBERT on EVERY chunk
    3. Collect (keyword, score) from all chunks
    4. Same keyword in multiple chunks → SUM scores (frequency boost)
    5. Normalize: final_score = total_score * log(1 + chunk_count)
    6. Sort by aggregated score → top keywords cover WHOLE document
    7. Deduplicate similar/overlapping keywords
    8. Supplement with legal term matching if still short
    """
    import math

    if not BERT_LOADED:
        text_lower = text.lower()
        return [t for t in LEGAL_TERMS if t in text_lower][:count]

    # Step 1: Chunk full document
    chunks = chunk_text(text, chunk_size=600, overlap=50)

    # Step 2 & 3: Run KeyBERT on every chunk
    scores_map  = {}   # keyword → total score
    chunk_count = {}   # keyword → how many chunks it appeared in
    bad = re.compile(r'\d|^.{1,2}$|^[^a-z]')

    print(f"[BERT] Processing {len(chunks)} chunks...")
    for i, chunk in enumerate(chunks):
        try:
            raw = KW_MODEL.extract_keywords(
                chunk,
                keyphrase_ngram_range=(1, 3),
                stop_words='english',
                use_mmr=True,
                diversity=0.6,
                top_n=12
            )
            for kw, score in raw:
                kw_clean = kw.lower().strip()
                if bad.search(kw_clean):
                    continue
                # Reject OCR artifacts and archaic words
                if any(bl in kw_clean for bl in KEYWORD_BLACKLIST):
                    continue
                # Reject keywords with non-alpha chars
                if __import__('re').search(r'[^a-z\s\-]', kw_clean):
                    continue
                # Require at least one word of 5+ chars
                if not any(len(w) >= 5 for w in kw_clean.split()):
                    continue
                # Skip overly generic legal phrases
                generic = ['said', 'above', 'hereby', 'herein', 'thereof', 'whereof',
                           'called settle', 'impose additional', 'statement accounts',
                           'shall mean', 'means include', 'following terms']
                if any(g in kw_clean for g in generic):
                    continue
                # Must have at least one meaningful word (4+ chars)
                if not any(len(w) >= 4 for w in kw_clean.split()):
                    continue
                if kw_clean in scores_map:
                    scores_map[kw_clean]  += score
                    chunk_count[kw_clean] += 1
                else:
                    scores_map[kw_clean]  = score
                    chunk_count[kw_clean] = 1
        except Exception as e:
            print(f"[BERT] Chunk {i} failed: {e}")

    print(f"[BERT] Unique candidates found: {len(scores_map)}")

    # Step 4 & 5: Normalize — frequency across chunks boosts score
    # keyword seen in 5 chunks > keyword seen in 1 chunk
    normalized = {}
    for kw, total_score in scores_map.items():
        freq_boost      = math.log(1 + chunk_count[kw])
        normalized[kw]  = total_score * freq_boost

    # Step 6: Sort by aggregated score
    sorted_kws = sorted(normalized.items(), key=lambda x: x[1], reverse=True)
    print(f"[BERT] Top 10 candidates: {[k for k,v in sorted_kws[:10]]}")

    # Step 7: Deduplicate similar keywords
    # e.g. "share capital" and "capital shares" → keep higher scored
    final_keywords = []
    for kw, score in sorted_kws:
        is_duplicate = False
        for chosen in final_keywords:
            if kw in chosen or chosen in kw:
                is_duplicate = True
                break
            kw_words     = set(kw.split())
            chosen_words = set(chosen.split())
            if kw_words and chosen_words:
                overlap = len(kw_words & chosen_words)
                if overlap / max(len(kw_words), len(chosen_words)) > 0.5:
                    is_duplicate = True
                    break
        if not is_duplicate:
            final_keywords.append(kw)
        if len(final_keywords) >= count * 2:
            break

    # Step 8: Supplement with legal terms if needed
    if len(final_keywords) < count:
        text_lower = text.lower()
        for term in LEGAL_TERMS:
            if term in text_lower and term not in final_keywords:
                final_keywords.append(term)

    result = final_keywords[:count]
    # Ensure minimum 6 keywords using LEGAL_TERMS fallback
    if len(result) < 6:
        text_lower = text.lower()
        for term in LEGAL_TERMS:
            if term in text_lower and term not in result:
                result.append(term)
                if len(result) >= 6:
                    break
    print(f"[BERT] Final keywords: {result}")
    return result if result else ["legal", "document", "contract", "agreement", "liability", "claim"]


# ─────────────────────────────────────────────
# MAIN API ENDPOINT
# ─────────────────────────────────────────────

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    content   = await file.read()
    raw_text  = extract_text(content, file.content_type, file.filename)
    text      = clean_text(raw_text)

    word_count = len(text.split())
    read_time  = max(1, word_count // 200)

    # Step 1 — Detect document type (rule-based)
    doc_type = detect_document_type(text)

    # Step 2 — Header regex + QA + NER extract all fields
    #   Header regex → claimant, respondents (high precision)
    #   QA          → claimant, respondents (medium precision)
    #   NER         → claimant, respondents, venue, persons, orgs, locations
    #   Regex       → amount, dates, case number, law, interest rate
    fields = extract_all_fields(text, doc_type)

    # Step 3 — Build summary using extracted fields
    summary = build_summary(fields, doc_type, text, word_count)

    # Step 4 — Legal-BERT KeyBERT extracts keywords
    keywords = get_keywords(text, count=7)

    # Step 5 — Build extraction_warnings for UI
    warnings = []
    if not fields.get("claimant"):
        warnings.append({
            "field": "claimant",
            "message": "Claimant could not be identified from this document.",
            "severity": "high" if doc_type in ("arbitration_claim", "court_judgment", "cheque_bounce") else "low"
        })
    if not fields.get("respondents"):
        warnings.append({
            "field": "respondents",
            "message": "Respondent(s) could not be identified from this document.",
            "severity": "high" if doc_type in ("arbitration_claim", "court_judgment", "cheque_bounce") else "low"
        })
    if doc_type == "arbitration_claim" and not fields.get("venue"):
        warnings.append({
            "field": "venue",
            "message": "Arbitration venue not specified in the document.",
            "severity": "medium"
        })
    if doc_type == "arbitration_claim" and not fields.get("amount"):
        warnings.append({
            "field": "amount",
            "message": "Claim amount could not be identified.",
            "severity": "medium"
        })

    return JSONResponse({
        "summary":               summary,
        "keywords":              keywords,
        "word_count":            word_count,
        "read_time":             read_time,
        "document_type":         doc_type,
        "ner_used":              NER_LOADED,
        "bert_used":             BERT_LOADED,
        "t5_used":               T5_LOADED,
        "qa_used":               QA_LOADED,
        "fields":                fields,
        "extraction_warnings":   warnings,
        "extraction_successful": len(warnings) == 0,
    })


# Serve frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")
