import re
from app.config import settings

#PII_PATTERNS = [
#    re.compile(r"\b\d{10}\b"),  # rough phone / Aadhaar-like
#    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),  # email
#]
#BANNED_KEYWORDS = ["classified", "top secret", "credit card", "ssn", "password", "bank account"]

def validate_file(filename: str, file_bytes: bytes):
    ext = "." + filename.lower().rsplit(".", 1)[-1]
    if ext not in settings.ALLOWED_EXT:
        return False, "File type not allowed."

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        return False, f"File too large ({size_mb:.2f} MB)."

    return True, ""

"""def validate_content(text: str):
    hits = []
    for patt in PII_PATTERNS:
        for m in patt.finditer(text):
            hits.append(("PII", m.group(0)))

    for kw in BANNED_KEYWORDS:
        if kw.lower() in text.lower():
            hits.append(("BANNED_KEYWORD", kw))

    if hits:
        return False, hits
    return True, []"""
