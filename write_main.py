content = open('/home/tharani/justact_bert/main.py.backup').read()

# Fix 1: Add KEYWORD_BLACKLIST before LEGAL_TERMS
blacklist = '''
KEYWORD_BLACKLIST = {
    'witnesseth', 'inconsideration', 'oromission', 'huoiciat',
    'quarented', 'vaeinod', 'addreswitness', 'patullos', 'emered',
    'anyact', 'sd rey', 'hereinbefore', 'india huoiciat',
    'aforesaid', 'aforementioned', 'hereunder', 'whereof',
    'thereof', 'thence', 'hereby', 'herein', 'hereto',
    'said', 'above', 'charges dues', 'emered borrower',
}

'''
content = content.replace('\nLEGAL_TERMS = [', blacklist + '\nLEGAL_TERMS = [', 1)

# Fix 2: Add blacklist filter inside get_keywords after the 'bad' regex check
old = "                if bad.search(kw_clean):\n                    continue"
new = """                if bad.search(kw_clean):
                    continue
                # Reject OCR artifacts and archaic words
                if any(bl in kw_clean for bl in KEYWORD_BLACKLIST):
                    continue
                # Reject keywords with non-alpha chars
                if __import__('re').search(r'[^a-z\\s\\-]', kw_clean):
                    continue
                # Require at least one word of 5+ chars
                if not any(len(w) >= 5 for w in kw_clean.split()):
                    continue"""
content = content.replace(old, new, 1)

# Fix 3: Expand noise_patterns in clean_text
old_noise = "        'wx ', 'wr ', 'wm ', 'hz ', 'hx ', 'sz ',"
new_noise = """        'wx ', 'wr ', 'wm ', 'hz ', 'hx ', 'sz ',
        'addreswitness', 'witnesseth', 'inconsideration',
        'oromission', 'huoiciat', 'quarented', 'vaeinod',
        'ay ee waite', 'patullos', 'emered borrower',
        'sd rey', 'anyact', 'india huoiciat',"""
content = content.replace(old_noise, new_noise, 1)

open('/home/tharani/justact_bert/main.py', 'w').write(content)
print("Done! Verifying...")
result = content.count('KEYWORD_BLACKLIST')
print(f"KEYWORD_BLACKLIST appears {result} times — {'OK' if result >= 1 else 'FAILED'}")
