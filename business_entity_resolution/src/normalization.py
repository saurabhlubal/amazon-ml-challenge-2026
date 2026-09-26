"""
Normalization module for Business Entity Resolution.
Generates multiple useful representations while preserving raw text,
symbols like '+', and multilingual/combining characters (e.g. Devanagari, accents).
Supports domain/social artifact stripping, compact representations, and address component parsing.
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
    "pc": "",  # Professional Corporation
    "llc": "llc",
    "llp": "llp",
}

# Legal suffixes to exclude from compact core name
CORE_LEGAL_SUFFIXES = {
    "corp", "inc", "co", "ltd", "pvt", "llc", "llp", "pc", "gmbh", "sa", "sarl", "bv",
    "services", "service", "solutions", "solution", "group", "holdings", "enterprises",
    "enterprise", "company", "limited", "incorporated", "corporation",
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

# US State abbreviations mapping
US_STATE_MAP = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "new hampshire": "nh",
    "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn",
    "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
}

# Explicit punctuation to replace with spaces.
# NOTE: '+' is intentionally preserved for entities like 'B+ Retail Inc'.
PUNCT_CHARS = r"""!"#$%&'()*,-./:;<=>?@[\]^_`{|}~‘’“”–—·•…"""
PUNCT_TRANS = str.maketrans({c: " " for c in PUNCT_CHARS})
WHITESPACE_REGEX = re.compile(r"\s+", flags=re.UNICODE)
DIGITS_REGEX = re.compile(r"\b\d+\b", flags=re.UNICODE)
DOMAIN_SUFFIX_REGEX = re.compile(
    r"\.(?:com|net|org|in|co|io|biz|info|us|fr|gov|edu|ai|app|tech)(?:\.[a-z]{2})?$",
    flags=re.IGNORECASE
)


def clean_text(text: Any) -> str:
    """Basic unicode normalization, html unescaping, domain artifact removal, and whitespace cleaning."""
    if not text:
        return ""
    text = str(text)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()

    # Strip web domain / handle prefixes and suffixes
    if text.startswith("@"):
        text = text[1:]
    if text.startswith("http://") or text.startswith("https://"):
        text = re.sub(r"^https?://(?:www\.)?", "", text)
    elif text.startswith("www."):
        text = text[4:]
    text = DOMAIN_SUFFIX_REGEX.sub("", text)

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
        val = BUSINESS_ABBREVIATIONS.get(w, w)
        if val:
            expanded.extend(val.split())
    return " ".join(expanded)


def get_compact_name(norm_name_str: str) -> str:
    """Core name stripped of legal suffixes and spaces (e.g. 'Prime Money' -> 'primemoney')."""
    if not norm_name_str:
        return ""
    words = [w for w in norm_name_str.split() if w not in CORE_LEGAL_SUFFIXES]
    if not words:
        words = norm_name_str.split()
    return "".join(words)


def normalize_address(text: Any) -> str:
    """Normalize business address with street/unit abbreviations and state normalizations."""
    cleaned = clean_text(text)
    if not cleaned:
        return ""
    words = cleaned.split()
    expanded = []
    for w in words:
        w_addr = ADDRESS_ABBREVIATIONS.get(w, w)
        w_state = US_STATE_MAP.get(w_addr, w_addr)
        expanded.extend(w_state.split())
    return " ".join(expanded)


def get_token_signature(text: str) -> str:
    """Alphabetically sorted unique tokens to handle word reorderings."""
    if not text:
        return ""
    tokens = sorted(set(text.split()))
    return " ".join(tokens)


def extract_numbers(text: Any) -> Set[str]:
    """Extract numeric tokens stripped of leading zeros (e.g., '0017560' -> '17560')."""
    if not text:
        return set()
    raw_nums = DIGITS_REGEX.findall(str(text))
    clean_nums = set()
    for n in raw_nums:
        stripped = n.lstrip("0")
        clean_nums.add(stripped if stripped else "0")
    return clean_nums


def extract_postal_codes(numbers: Set[str]) -> Set[str]:
    """Extract likely postal / PIN codes (5-digit for US/France, 6-digit for India)."""
    return {n for n in numbers if len(n) in (5, 6)}


def extract_char_shingles(compact_name: str, n: int = 3) -> List[str]:
    """Extract distinct character 3-grams of compact name."""
    if not compact_name or len(compact_name) < n:
        return [compact_name] if compact_name else []
    shingles = set()
    for i in range(len(compact_name) - n + 1):
        shingles.add(compact_name[i:i + n])
    return sorted(shingles)


def extract_state(norm_addr_str: str) -> str:
    """Extract US state 2-letter abbreviation if present."""
    if not norm_addr_str:
        return ""
    tokens = set(norm_addr_str.split())
    # Check 2-letter states
    valid_states = set(US_STATE_MAP.values())
    for tok in tokens:
        if tok in valid_states:
            return tok
    return ""


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
    compact_name = get_compact_name(norm_name_str)

    name_tokens = norm_name_str.split() if norm_name_str else []
    addr_tokens = norm_addr_str.split() if norm_addr_str else []
    clean_nums = extract_numbers(raw_address)
    postal_codes = extract_postal_codes(clean_nums)
    char_shingles = extract_char_shingles(compact_name, 3)
    state = extract_state(norm_addr_str)

    # Street number: first number from the address
    street_num = ""
    for w in norm_addr_str.split():
        w_clean = w.lstrip("0")
        if w_clean in clean_nums and len(w_clean) <= 6:
            street_num = w_clean
            break

    return {
        "raw": record,
        "entity_id": entity_id,
        "country": country,
        "business_name": norm_name_str,
        "business_address": norm_addr_str,
        "compact_name": compact_name,
        "raw_business_name": raw_name,
        "raw_business_address": raw_address,
        "name_tokens": name_tokens,
        "name_tokens_set": set(name_tokens),
        "name_signature": get_token_signature(norm_name_str),
        "address_tokens": addr_tokens,
        "address_tokens_set": set(addr_tokens),
        "address_signature": get_token_signature(norm_addr_str),
        "address_numbers": clean_nums,
        "postal_codes": postal_codes,
        "street_number": street_num,
        "char_shingles": char_shingles,
        "state": state,
    }