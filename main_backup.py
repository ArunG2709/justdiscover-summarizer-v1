from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import pdfplumber
import docx
import re
import io

app = FastAPI()

# ─────────────────────────────────────────
# TEXT EXTRACTION
# ─────────────────────────────────────────

def extract_text(content: bytes, content_type: str, filename: str) -> str:
    try:
        if filename.lower().endswith(".pdf") or content_type == "application/pdf":
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                pages = []
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        pages.append(t)
                return "\n".join(pages)
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
    for line in lines:
        if len(line.strip()) < 3:
            continue
        digit_count = sum(c.isdigit() for c in line)
        if len(line) > 0 and digit_count / len(line) > 0.6:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


# ─────────────────────────────────────────
# DOCUMENT TYPE DETECTION
# ─────────────────────────────────────────

def detect_document_type(text: str) -> str:
    text_lower = text.lower()

    # Arbitration / NPA Claim Statement
    if ("arbitration" in text_lower or "claimant" in text_lower) and \
       ("respondent" in text_lower or "loan" in text_lower or "npa" in text_lower):
        return "arbitration_claim"

    # Companies Act / Legislation
    if "companies act" in text_lower and ("chapter" in text_lower or "section" in text_lower):
        return "companies_act"

    # Income Tax / GST
    if "income tax" in text_lower or "assessment year" in text_lower or "gst" in text_lower:
        return "tax_document"

    # Court Judgment
    if ("versus" in text_lower or " v. " in text_lower) and \
       ("honourable" in text_lower or "judgment" in text_lower or "petitioner" in text_lower):
        return "court_judgment"

    # Legal Notice
    if "legal notice" in text_lower or "notice is hereby given" in text_lower:
        return "legal_notice"

    # Contract / Agreement
    if ("agreement" in text_lower or "contract" in text_lower) and \
       ("party of the first part" in text_lower or "hereinafter referred to as" in text_lower):
        return "contract"

    # FIR / Police Document
    if "first information report" in text_lower or "f.i.r" in text_lower:
        return "fir"

    # Cheque Bounce / Section 138
    if "section 138" in text_lower or "negotiable instruments act" in text_lower:
        return "cheque_bounce"

    return "general_legal"


# ─────────────────────────────────────────
# RULE-BASED FIELD EXTRACTORS (for arbitration)
# ─────────────────────────────────────────

def extract_claimant(text):
    patterns = [
        r'between\s+(.+?)\s+(?:\.{3,}|…)\s*claimant',
        r'claimant[:\s]+([A-Z][a-zA-Z\s]+(?:Limited|Ltd|LLP|Finance|Bank)?)',
        r'([A-Z][a-zA-Z\s]+(?:Limited|Ltd|Finance|Bank))\s+\.{3,}\s*(?:Claimant|CLAIMANT)',
        r'(?:claimant above named is|claimant is)\s+(.+?)(?:\.|,|\n)',
        r'M[/\.]s\.?\s+([A-Z][a-zA-Z\s]+(?:Limited|Ltd)?)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            name = match.group(1).strip()
            name = re.sub(r'\s+', ' ', name)
            name = name.strip('.,')
            if 3 < len(name) < 80:
                return name
    return None


def extract_respondents(text):
    respondents = []
    patterns = [
        r'(?:1st|first)\s+respondent[:\s]+(.+?)(?:S/O|W/O|D/O|,|\n|\.)',
        r'(?:2nd|second)\s+respondent[:\s]+(.+?)(?:S/O|W/O|D/O|,|\n|\.)',
        r'(?:3rd|third)\s+respondent[:\s]+(.+?)(?:S/O|W/O|D/O|,|\n|\.)',
        r'respondent(?:s)?[:\s]+(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?)?\s*([A-Z][a-zA-Z\s]+)',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            name = match.strip().strip('.,')
            if 2 < len(name) < 60:
                respondents.append(name)
    respondents = list(dict.fromkeys(respondents))
    return respondents[:3] if respondents else []


def extract_case_number(text):
    patterns = [
        r'(?:Arbitration Case No|Case No|Ref No|Reference No)[.:\s]+([A-Z0-9/\-]+)',
        r'([A-Z]{1,5}\d{6,}/[A-Z]{1,5}/\d{4,}/[A-Z]{1,5}/\d{4})',
        r'([A-Z]\d{9,}/[A-Z]+/\d+/[A-Z]+/\d{4})',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def extract_amount(text):
    patterns = [
        r'sum\s+of\s+Rs\.?\s*([\d,]+(?:\.\d{1,2})?)',
        r'Rs\.?\s*([\d,]+(?:\.\d{1,2})?)\s+(?:ps\s+)?(?:is\s+)?due',
        r'NET\s+Amount[:\s]+Rs\.?\s*([\d,]+(?:\.\d{1,2})?)',
        r'total\s+(?:claim\s+)?amount[:\s]+Rs\.?\s*([\d,]+(?:\.\d{1,2})?)',
        r'outstanding\s+amount[:\s]+Rs\.?\s*([\d,]+(?:\.\d{1,2})?)',
        r'amount\s+due[:\s]+Rs\.?\s*([\d,]+(?:\.\d{1,2})?)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            amt = match.group(1).replace(',', '')
            try:
                float(amt)
                return f"Rs.{match.group(1)}"
            except:
                continue
    return None


def extract_law(text):
    patterns = [
        r'(?:under|pursuant to|in terms of)\s+the\s+(Arbitration\s+(?:and|&)\s+Conciliation\s+Act[,\s]+\d{4})',
        r'(Arbitration\s+(?:and|&)\s+Conciliation\s+Act[,\s]+\d{4})',
        r'(Companies\s+Act[,\s]+\d{4})',
        r'(Negotiable\s+Instruments\s+Act[,\s]+\d{4})',
        r'(Indian\s+Contract\s+Act[,\s]+\d{4})',
        r'(Transfer\s+of\s+Property\s+Act[,\s]+\d{4})',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def extract_venue(text):
    patterns = [
        r'(?:venue|arbitration\s+at|seat\s+of\s+arbitration)[:\s]+([A-Z][a-zA-Z\s]+?)(?:\.|,|\n)',
        r'(?:place\s+of\s+arbitration)[:\s]+([A-Z][a-zA-Z]+)',
        r'arbitration\s+(?:proceedings?\s+)?(?:at|in)\s+([A-Z][a-zA-Z]+)',
    ]
    cities = ["Chennai", "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Kolkata",
              "Pune", "Ahmedabad", "Coimbatore", "Madurai"]
    for city in cities:
        if city.lower() in text.lower():
            if "venue" in text.lower() or "arbitration" in text.lower():
                return city
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def extract_dates(text):
    dates = {}
    loan_patterns = [
        r'loan\s+(?:was\s+)?(?:availed|taken|dated?)[:\s]+(\d{1,2}[-/]\w{3}[-/]\d{4}|\d{1,2}[-/]\d{1,2}[-/]\d{4})',
        r'loan\s+agreement\s+(?:dated?|on)[:\s]+(\d{1,2}[-/]\w{3}[-/]\d{4})',
        r'date\s+of\s+loan[:\s]+(\d{1,2}[-/]\w{3}[-/]\d{4})',
    ]
    default_patterns = [
        r'(?:default(?:ed)?|defaulting)\s+(?:from|on|since)[:\s]+(\d{1,2}[-/]\w{3}[-/]\d{4}|\d{1,2}[-/]\d{1,2}[-/]\d{4})',
        r'due\s+(?:from|on|since)[:\s]+(\d{1,2}[-/]\w{3}[-/]\d{4})',
        r'(?:payments?\s+due\s+from)[:\s]+(\d{1,2}[-/]\w{3}[-/]\d{4})',
    ]
    for p in loan_patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            dates['loan_date'] = m.group(1)
            break
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
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return f"{match.group(1)}% p.a."
    return None


# ─────────────────────────────────────────
# SUMMARY BUILDERS PER DOCUMENT TYPE
# ─────────────────────────────────────────

def build_summary_arbitration(fields):
    parts = []
    claimant = fields.get('claimant', 'The claimant')
    respondents = fields.get('respondents', [])
    respondent_str = ', '.join(respondents) if respondents else 'the respondent'
    law = fields.get('law', 'the Arbitration and Conciliation Act, 1996')

    parts.append(f"{claimant} has filed an arbitration claim against {respondent_str} under the {law}.")

    if fields.get('case_number'):
        parts.append(f"The case number is {fields['case_number']}.")

    dates = fields.get('dates', {})
    if dates.get('loan_date'):
        parts.append(f"A loan agreement was entered on {dates['loan_date']}, and the respondent defaulted on payments due from {dates.get('default_date', 'a subsequent date')}.")
    elif dates.get('default_date'):
        parts.append(f"The respondent defaulted on payments due from {dates['default_date']}.")

    if fields.get('amount'):
        parts.append(f"The total claim amount outstanding is {fields['amount']}.")

    if fields.get('venue'):
        parts.append(f"The arbitration venue is {fields['venue']}.")

    if fields.get('interest_rate'):
        parts.append(f"The claimant seeks recovery of the outstanding amount along with interest at {fields['interest_rate']} and costs of proceedings.")

    return " ".join(parts)


def build_summary_companies_act(text, word_count):
    chapters = len(re.findall(r'\bCHAPTER\s+[IVXLC]+\b', text, re.IGNORECASE))
    sections = len(re.findall(r'\bSection\s+\d+\b', text, re.IGNORECASE))
    schedules = len(re.findall(r'\bSCHEDULE\s+[IVXLC]+\b', text, re.IGNORECASE))

    summary = f"This is the Companies Act, 2013 — the primary legislation governing company law in India. "
    summary += f"The Act contains {word_count:,} words"
    if chapters:
        summary += f", organized across {chapters} chapters"
    if sections:
        summary += f" with {sections}+ sections"
    if schedules:
        summary += f" and {schedules} schedules"
    summary += ". "
    summary += "It covers company incorporation, share capital, management, audit, directors, meetings, "
    summary += "corporate social responsibility, winding up, and related matters. "
    summary += "The Act replaced the Companies Act, 1956 and is administered by the Ministry of Corporate Affairs."
    return summary


def build_summary_tax(text, word_count):
    year_match = re.search(r'assessment\s+year[:\s]+(\d{4}-\d{2,4})', text, re.IGNORECASE)
    assessee_match = re.search(r'(?:assessee|taxpayer)[:\s]+([A-Z][a-zA-Z\s]+?)(?:\.|,|\n)', text, re.IGNORECASE)

    summary = "This is a tax-related legal document. "
    if assessee_match:
        summary += f"It pertains to {assessee_match.group(1).strip()}. "
    if year_match:
        summary += f"The assessment year is {year_match.group(1)}. "
    summary += f"The document contains {word_count:,} words covering tax assessment, computation of income, deductions, and related tax matters."
    return summary


def build_summary_court_judgment(text, word_count):
    petitioner = None
    respondent = None

    pet_match = re.search(r'(?:petitioner|appellant)[:\s]+([A-Z][a-zA-Z\s]+?)(?:\s+v\.|\s+vs\.|\n)', text, re.IGNORECASE)
    res_match = re.search(r'(?:respondent|appellee)[:\s]+([A-Z][a-zA-Z\s]+?)(?:\.|,|\n)', text, re.IGNORECASE)
    if pet_match:
        petitioner = pet_match.group(1).strip()
    if res_match:
        respondent = res_match.group(1).strip()

    summary = "This is a court judgment/order. "
    if petitioner and respondent:
        summary += f"The case involves {petitioner} versus {respondent}. "
    summary += f"The document contains {word_count:,} words covering the facts, arguments, legal issues, and the court's decision in the matter."
    return summary


def build_summary_legal_notice(text, word_count):
    summary = "This is a legal notice. "
    law = extract_law(text)
    if law:
        summary += f"The notice is issued under {law}. "
    amount = extract_amount(text)
    if amount:
        summary += f"A claim amount of {amount} has been demanded. "
    summary += f"The document contains {word_count:,} words and formally communicates legal rights, claims, or demands to the recipient."
    return summary


def build_summary_contract(text, word_count):
    summary = "This is a legal contract/agreement. "
    parties = re.findall(r'(?:party of the first part|hereinafter referred to as)[:\s"]+([A-Z][a-zA-Z\s]+?)(?:"|,|\)|\.)', text, re.IGNORECASE)
    if parties:
        summary += f"The agreement involves {', '.join(parties[:2])}. "
    summary += f"The document contains {word_count:,} words covering the terms, conditions, rights, and obligations of the contracting parties."
    return summary


def build_summary_cheque_bounce(text, word_count):
    summary = "This is a cheque dishonour case under Section 138 of the Negotiable Instruments Act, 1881. "
    amount = extract_amount(text)
    if amount:
        summary += f"The dishonoured cheque amount is {amount}. "
    summary += f"The document contains {word_count:,} words and covers the complaint, facts, and legal proceedings related to cheque dishonour."
    return summary


def build_summary_general(text, word_count):
    # Extract any available fields
    claimant = extract_claimant(text)
    amount = extract_amount(text)
    law = extract_law(text)

    summary = "This is a legal document. "
    if claimant:
        summary += f"It involves {claimant}. "
    if law:
        summary += f"It references {law}. "
    if amount:
        summary += f"An amount of {amount} is mentioned. "
    summary += f"The document contains {word_count:,} words. Keywords have been extracted using Legal-BERT for further analysis."
    return summary


def build_summary(fields, doc_type, text, word_count):
    if doc_type == "arbitration_claim":
        return build_summary_arbitration(fields)
    elif doc_type == "companies_act":
        return build_summary_companies_act(text, word_count)
    elif doc_type == "tax_document":
        return build_summary_tax(text, word_count)
    elif doc_type == "court_judgment":
        return build_summary_court_judgment(text, word_count)
    elif doc_type == "legal_notice":
        return build_summary_legal_notice(text, word_count)
    elif doc_type == "contract":
        return build_summary_contract(text, word_count)
    elif doc_type == "cheque_bounce":
        return build_summary_cheque_bounce(text, word_count)
    else:
        return build_summary_general(text, word_count)


# ─────────────────────────────────────────
# KEYWORD EXTRACTION (Legal-BERT / KeyBERT)
# ─────────────────────────────────────────

LEGAL_TERMS = [
    "arbitration", "hypothecation", "guarantor", "liquidated damages",
    "loan agreement", "default", "instalment", "claimant", "respondent",
    "conciliation", "indemnity", "mortgage", "lien", "subrogation",
    "jurisdiction", "injunction", "affidavit", "summons", "decree",
    "encumbrance", "pledge", "collateral", "promissory note", "debenture",
    "insolvency", "liquidation", "winding up", "foreclosure", "caveat",
    "dividend", "share capital", "memorandum", "articles of association",
    "incorporation", "dissolution", "amalgamation", "merger", "acquisition",
    "litigation", "adjudication", "tribunal", "appellate", "statutory",
    "compliance", "outstanding", "liability", "asset", "creditor", "debtor",
    "penalty", "forfeiture", "recovery", "settlement", "mediation",
]

def get_keywords(text: str, count: int = 7):
    text_lower = text.lower()
    found = []
    for term in LEGAL_TERMS:
        if term in text_lower and term not in found:
            found.append(term)
        if len(found) >= count:
            break

    if len(found) >= count:
        return found[:count]

    # Fallback: KeyBERT with Legal-BERT
    try:
        from keybert import KeyBERT
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer('nlpaueb/legal-bert-base-uncased')
        kw_model = KeyBERT(model=model)
        mid = len(text) // 4
        sample = text[mid: mid + 3000]

        keywords = kw_model.extract_keywords(
            sample,
            keyphrase_ngram_range=(1, 2),
            stop_words='english',
            top_n=count * 2
        )

        bad_patterns = re.compile(r'\d|^.{1,2}$|^[^a-z]')
        for kw, score in keywords:
            kw_clean = kw.lower().strip()
            if not bad_patterns.search(kw_clean) and kw_clean not in found:
                found.append(kw_clean)
            if len(found) >= count:
                break
    except Exception:
        pass

    return found[:count] if found else ["legal", "document", "analysis"]


# ─────────────────────────────────────────
# MAIN API ENDPOINT
# ─────────────────────────────────────────

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    content = await file.read()
    raw_text = extract_text(content, file.content_type, file.filename)
    text = clean_text(raw_text)

    word_count = len(text.split())
    read_time = max(1, word_count // 200)

    # Detect document type
    doc_type = detect_document_type(text)

    # Extract fields (mainly for arbitration; harmless for others)
    fields = {}
    if doc_type == "arbitration_claim":
        fields['claimant'] = extract_claimant(text)
        fields['respondents'] = extract_respondents(text)
        fields['case_number'] = extract_case_number(text)
        fields['amount'] = extract_amount(text)
        fields['law'] = extract_law(text)
        fields['venue'] = extract_venue(text)
        fields['dates'] = extract_dates(text)
        fields['interest_rate'] = extract_interest_rate(text)

    # Build summary based on document type
    summary = build_summary(fields, doc_type, text, word_count)

    # Extract keywords
    keywords = get_keywords(text, count=7)

    return JSONResponse({
        "summary": summary,
        "keywords": keywords,
        "word_count": word_count,
        "read_time": read_time,
        "document_type": doc_type,
        "fields": fields
    })


# Serve frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")
