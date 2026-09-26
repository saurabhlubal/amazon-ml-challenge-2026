"""
Normalization module for Business Entity Resolution.
Generates multiple useful representations while preserving raw text,
symbols like '+', and multilingual/combining characters (e.g. Devanagari, accents).
"""

import re
import html
import unicodedata
from typing import Dict, Any, List, Set


# Common business legal suffixes & entity terms
BUSINESS_ABBREVIATIONS = {
    "corporation": "corp",
    "incorporated": "inc",
    "company": "co",
    "limited": "ltd",
    "private": "pvt",
    "privatelimited": "pvt ltd",
    "public": "pub",
    "enterprises": "ent",
    "enterprise": "ent",
    "solutions": "sol",
    "services": "svc",
    "service": "svc",
    "technologies": "tech",
    "technology": "tech",
    "international": "intl",
    "associates": "assoc",
    "industries": "ind",
    "industry": "ind",
    "consultants": "consult",
    "consultancy": "consult",
    "marketing": "mktg",
    "properties": "prop",
    "property": "prop",
}

# Common address term abbreviations
ADDRESS_ABBREVIATIONS = {
    "road": "rd",
    "street": "st",
    "avenue": "ave",
    "boulevard": "blvd",
    "highway": "hwy",
    "drive": "dr",
    "lane": "ln",
    "court": "ct",
    "parkway": "pkwy",
    "place": "pl",
    "square": "sq",
    "apartment": "apt",
    "suite": "ste",
    "floor": "fl",
    "building": "bldg",
    "extension": "ext",
    "opposite": "opp",
    "near": "nr",
    "circle": "cir",
    "post": "po",
}

# Explicit punctuation to replace with spaces.
# NOTE: '+' is intentionally preserved for entities like 'B+ Retail Inc'.
# Combining marks (Devanagari matras, accents) and unicode letters are fully preserved.
PUNCT_CHARS = r"""!"#$%&'()*,-./:;<=>?@[\]^_`{|}~‘’“”–—·•…"""
PUNCT_TRANS = str.maketrans({c: " " for c in PUNCT_CHARS})
WHITESPACE_REGEX = re.compile(r"\s+", flags=re.UNICODE)
DIGITS_REGEX = re.compile(r"\b\d+\b", flags=re.UNICODE)


def clean_text(text: Any) -> str:
    """Basic unicode normalization, html unescaping, and whitespace cleaning."""
    if not text:
        return ""
    text = str(text)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = text.translate(PUNCT_TRANS)
    text = WHITESPACE_REGEX.sub(" ", text).strip()
    return text


def normalize_name(text: Any) -> str:
    """Normalize business name with suffix abbreviations."""
    cleaned = clean_text(text)
    if not cleaned:
        return ""
    words = cleaned.split()
    expanded = []
    for w in words:
        expanded.extend(BUSINESS_ABBREVIATIONS.get(w, w).split())
    return " ".join(expanded)


def normalize_address(text: Any) -> str:
    """Normalize business address with street/unit abbreviations."""
    cleaned = clean_text(text)
    if not cleaned:
        return ""
    words = cleaned.split()
    expanded = []
    for w in words:
        expanded.extend(ADDRESS_ABBREVIATIONS.get(w, w).split())
    return " ".join(expanded)


def get_token_signature(text: str) -> str:
    """Alphabetically sorted unique tokens to handle word reorderings."""
    if not text:
        return ""
    tokens = sorted(set(text.split()))
    return " ".join(tokens)


def extract_numbers(text: Any) -> Set[str]:
    """Extract numeric tokens (e.g., street numbers, postal codes, unit numbers)."""
    if not text:
        return set()
    return set(DIGITS_REGEX.findall(str(text)))


def normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize one business entity, producing multiple representations.

    Parameters
    ----------
    record : dict
        Raw entity record containing:
        entity_id, business_name, business_address, country

    Returns
    -------
    dict
        Multiple normalized representations preserving raw fields and token structures.
    """
    entity_id = record.get("entity_id", "")
    raw_name = record.get("business_name", "")
    raw_address = record.get("business_address", "")
    country = str(record.get("country", "")).strip().upper()

    norm_name_str = normalize_name(raw_name)
    norm_addr_str = normalize_address(raw_address)

    name_tokens = norm_name_str.split() if norm_name_str else []
    addr_tokens = norm_addr_str.split() if norm_addr_str else []

    return {
        "raw": record,
        "entity_id": entity_id,
        "country": country,
        "business_name": norm_name_str,
        "business_address": norm_addr_str,
        "raw_business_name": raw_name,
        "raw_business_address": raw_address,
        "name_tokens": name_tokens,
        "name_tokens_set": set(name_tokens),
        "name_signature": get_token_signature(norm_name_str),
        "address_tokens": addr_tokens,
        "address_tokens_set": set(addr_tokens),
        "address_signature": get_token_signature(norm_addr_str),
        "address_numbers": extract_numbers(raw_address),
    }