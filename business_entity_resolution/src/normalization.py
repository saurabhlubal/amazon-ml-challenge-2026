"""
Text and Entity Normalization for Business Entity Resolution.

Handles:
- Unicode NFKC normalization and mojibake recovery
- Latin diacritic / accent stripping
- Multilingual Indic / Brahmic transliteration (Devanagari, Gujarati, Bengali, Odia,
  Gurmukhi, Tamil, Telugu, Kannada, Malayalam)
- Domain / URL root extraction from business names
- Common legal entity suffix normalization and stripping
- Address component normalization (street types, locality, hyphenated/composite numbers)
- Leetspeak and noisy symbol handling
"""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple

# Pre-compiled regular expressions for speed
RE_WHITESPACE = re.compile(r"\s+")
RE_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
RE_DIGITS = re.compile(r"\d+")
RE_DOMAIN = re.compile(r"([a-z0-9]+)\.(?:com|org|net|in|co|io|fr|gov|edu|biz|info)", re.IGNORECASE)

# Leetspeak / social handle character substitution table
LEET_MAP = str.maketrans({
    "@": "a",
    "0": "o",
    "1": "i",
    "3": "e",
    "$": "s",
    "5": "s",
    "7": "t",
})

# Common business legal suffixes (US, India, France, global)
LEGAL_SUFFIXES: Set[str] = {
    "corp", "corporation", "inc", "incorporated", "co", "company",
    "ltd", "limited", "pvt", "private", "llc", "llp", "lp", "plc",
    "sa", "sarl", "sas", "sasu", "sci", "eurl", "snc", "gmbh", "bv",
    "proprietorship", "enterprises", "enterprise", "associates", "partners"
}

# Legal entity substitution to canonical abbreviations
LEGAL_ENTITY_MAP: Dict[str, str] = {
    "corporation": "corp",
    "incorporated": "inc",
    "company": "co",
    "limited": "ltd",
    "private": "pvt",
    "privatelimited": "pvt ltd",
    "limitedliabilitycompany": "llc",
    "limitedliabilitypartnership": "llp",
    "societeanonyme": "sa",
    "societeparactions": "sas",
    "societearesponsabilite": "sarl",
}

# Street and address type standardizations
STREET_MAP: Dict[str, str] = {
    "street": "st",
    "road": "rd",
    "avenue": "ave",
    "boulevard": "blvd",
    "drive": "dr",
    "lane": "ln",
    "court": "ct",
    "circle": "cir",
    "highway": "hwy",
    "expressway": "expy",
    "parkway": "pkwy",
    "place": "pl",
    "square": "sq",
    "terrace": "ter",
    "suite": "ste",
    "apartment": "apt",
    "building": "bldg",
    "floor": "fl",
    "room": "rm",
    "sector": "sec",
    "phase": "ph",
    "plot": "plt",
    "nagar": "ngr",
    "colony": "clny",
    "marg": "mrg",
    "near": "nr",
    "opposite": "opp",
    "behind": "bhnd",
}

# Generic address tokens to exclude from locality indexing
GENERIC_ADDR_TOKENS: Set[str] = {
    "road", "rd", "street", "st", "avenue", "ave", "drive", "dr", "lane", "ln",
    "court", "ct", "suite", "ste", "floor", "fl", "north", "south", "east", "west",
    "near", "nr", "opposite", "opp", "behind", "bhnd", "phase", "ph", "sector", "sec",
    "nagar", "ngr", "colony", "clny", "city", "plot", "plt", "building", "bldg",
    "state", "india", "united", "states", "france", "room", "dept", "unit"
}

# Brahmic / Indic character mapping to phonetic Latin
DEVA_TO_LATIN: Dict[str, str] = {
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
    'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
    'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
    'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
    'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
    'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'श': 'sh', 'ष': 'sh', 'स': 's', 'ह': 'h',
    'क़': 'q', 'ख़': 'kh', 'ग़': 'gh', 'ज़': 'z', 'फ़': 'f',
    'ा': 'a', 'ि': 'i', 'ी': 'i', 'ु': 'u', 'ू': 'u', 'ृ': 'ri',
    'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au', 'ं': 'n', 'ः': 'h',
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo',
    'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au'
}

# Common transliterated Indian business vocabulary
COMMON_INDIC_WORDS: Dict[str, str] = {
    'प्राइवेट': 'pvt', 'लिमिटेड': 'ltd', 'प्रा': 'pvt', 'लि': 'ltd',
    'एलएलपी': 'llp', 'कंपनी': 'co', 'इन्वेस्टमेंट': 'investment',
    'डेवलपर्स': 'developers', 'फूड': 'food', 'वेंचर्स': 'ventures',
    'मार्केटिंग': 'marketing', 'प्रॉपर्टीज': 'properties',
    'कंस्ट्रक्शंस': 'constructions', 'टेक्नोलॉजीज': 'technologies',
    'एंटरप्राइजेज': 'enterprises', 'होटल': 'hotel', 'रियल': 'real',
    'मॉडर्न': 'modern', 'फर्स्ट': 'first', 'ग्लोबल': 'global',
    'अल्फा': 'alpha', 'अल': 'al', 'जैन': 'jain', 'लक्ष्मी': 'laxmi',
    'राम': 'ram', 'आदित्य': 'aditya', 'सन': 'sun', 'एग्रो': 'agro',
    'इन्फ्रा': 'infra', 'एस्टेट': 'estate', 'शक्ति': 'shakti',
    'इंडस्ट्रीज': 'industries', 'ट्रेडर्स': 'traders', 'सन्स': 'sons',
    'ड्रिम': 'dream', 'কনস্ট্রাকশন': 'construction', 'প্রাইভেট': 'pvt',
    'লিমিটেড': 'ltd', 'ইন্ডাস্ট্রিজ': 'industries'
}

# Mojibake correction map
MOJIBAKE_MAP: Dict[str, str] = {
    "Ã©": "é", "Ã¨": "è", "Ã ": "à", "Ã¢": "â", "Ã®": "î", "Ã¯": "ï",
    "Ã´": "ô", "Ã¹": "ù", "Ã»": "û", "Ã§": "ç", "â€™": "'", "â€œ": '"',
    "â€": '"', "â€“": "-", "â€”": "-", "â€¦": "...", "\ufffd": " "
}


def repair_mojibake(text: str) -> str:
    """Repair common mojibake UTF-8 byte sequences decoded as Latin-1."""
    if not text:
        return ""
    for bad, good in MOJIBAKE_MAP.items():
        if bad in text:
            text = text.replace(bad, good)
    return text


def strip_latin_accents(text: str) -> str:
    """
    Strip combining diacritical marks from Latin characters (e.g., é -> e).
    Preserves Indic characters, Chinese, and non-combining letters.
    """
    if not text:
        return ""
    chars = []
    for c in unicodedata.normalize("NFKD", text):
        if "\u0300" <= c <= "\u036f":
            continue
        chars.append(c)
    return "".join(chars)


def transliterate_indic(text: str) -> str:
    """
    Phonetically transliterates any Brahmic/Indic script into Latin English.
    Supports Devanagari, Gujarati, Bengali, Gurmukhi, Odia, Tamil, Telugu,
    Kannada, and Malayalam via Brahmic block alignment.
    """
    if not text:
        return ""
    # Map Brahmic Indic codepoints (0x0980-0x0D7F) to standard Devanagari block (0x0900-0x097F)
    chars = []
    has_indic = False
    for c in text:
        code = ord(c)
        if 0x0900 <= code <= 0x0D7F:
            has_indic = True
            if code >= 0x0980:
                offset = code % 0x80
                chars.append(chr(0x0900 + offset))
            else:
                chars.append(c)
        else:
            chars.append(c)

    if not has_indic:
        return text

    deva_text = "".join(chars)
    # Replace standard business terms
    for term, rep in COMMON_INDIC_WORDS.items():
        if term in deva_text:
            deva_text = deva_text.replace(term, rep)

    # Character-by-character phonetic Latin mapping
    res = []
    i = 0
    n = len(deva_text)
    while i < n:
        c = deva_text[i]
        if c in DEVA_TO_LATIN:
            val = DEVA_TO_LATIN[c]
            # Virama check (halant)
            if i + 1 < n and deva_text[i + 1] == '\u094d':
                res.append(val)
                i += 2
                continue
            # Matra check
            elif i + 1 < n and deva_text[i + 1] in [
                'ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'े', 'ै', 'ो', 'ौ', 'ं', 'ः'
            ]:
                res.append(val)
                i += 1
                continue
            else:
                res.append(val)
        else:
            if c != '\u094d':
                res.append(c)
        i += 1

    return "".join(res)


def clean_text(text: Optional[str]) -> str:
    """
    Unicode-safe string standardization:
    1. Unescape HTML entities (&amp;, &quot;, etc.)
    2. Repair mojibake
    3. Normalize Unicode (NFKC)
    4. Strip Latin accents (diacritics)
    5. Transliterate Indic characters
    6. Convert punctuation to whitespace
    7. Lowercase and collapse whitespace
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    text = html.unescape(text)
    text = repair_mojibake(text)
    text = unicodedata.normalize("NFKC", text)
    text = transliterate_indic(text)
    text = strip_latin_accents(text)
    text = RE_PUNCT.sub(" ", text.lower())
    return RE_WHITESPACE.sub(" ", text).strip()


def extract_domain_root(raw_name: str) -> Optional[str]:
    """
    Extracts core second-level domain name from business names containing URLs
    or social handles (e.g., 'bryansquare.com' -> 'bryansquare', 'kbmresearch.c0m' -> 'kbmresearch').
    """
    if not raw_name:
        return None
    # Map leetspeak for domain parsing
    norm_dom = raw_name.lower().translate(LEET_MAP)
    m = RE_DOMAIN.search(norm_dom)
    if m:
        root = m.group(1).strip()
        if len(root) >= 3:
            return root
    return None


def clean_digit(d: str) -> str:
    """Strip leading zeros from digit strings (e.g. '007' -> '7', '0356' -> '356')."""
    s = d.lstrip("0")
    return s if s else "0"


def normalize_business_name(raw_name: str) -> Dict[str, Any]:
    """
    Comprehensive business name normalization.

    Returns:
    - clean_name: fully cleaned unicode string
    - core_name: name with legal entity suffixes removed
    - tokens: list of clean word tokens
    - core_tokens: set of tokens without legal entity words
    - sorted_core: string of first 3 sorted core tokens
    - compact_forms: set of compact alphanumeric strings (for domain/composite matching)
    - domain_root: extracted domain root if URL present
    """
    raw_str = "" if raw_name is None else str(raw_name)
    domain_root = extract_domain_root(raw_str)

    # Strip leading handle symbols
    clean_input = raw_str
    if clean_input.startswith("@") or clean_input.startswith("#"):
        clean_input = clean_input[1:]

    clean_n = clean_text(clean_input)
    leet_n = clean_n.translate(LEET_MAP)

    tokens = clean_n.split()
    leet_tokens = leet_n.split()

    core_tokens = [t for t in tokens if t not in LEGAL_SUFFIXES]
    leet_core_tokens = [t for t in leet_tokens if t not in LEGAL_SUFFIXES]

    core_name = " ".join(core_tokens)
    compact_core = "".join(core_tokens)
    compact_full = "".join(tokens)
    compact_leet = "".join(leet_core_tokens)
    first2_compact = "".join(core_tokens[:2]) if len(core_tokens) >= 2 else None
    sorted_core = " ".join(sorted(core_tokens[:3])) if core_tokens else ""

    compact_forms: Set[str] = {compact_core, compact_full, compact_leet}
    if domain_root:
        compact_forms.add(domain_root)
    if first2_compact:
        compact_forms.add(first2_compact)
    compact_forms = {c for c in compact_forms if len(c) >= 3}

    all_tokens = set(core_tokens) | set(leet_core_tokens)

    return {
        "raw_name": raw_str,
        "clean_name": clean_n,
        "core_name": core_name,
        "tokens": tokens,
        "core_tokens": all_tokens,
        "sorted_core": sorted_core,
        "compact_forms": compact_forms,
        "domain_root": domain_root,
    }


def normalize_address(raw_address: str) -> Dict[str, Any]:
    """
    Comprehensive business address normalization.

    Returns:
    - clean_address: fully cleaned address string
    - tokens: list of clean word tokens
    - digits: list of normalized digit strings (leading zeros stripped)
    - first_digit: primary building/house number
    - second_digit: secondary building/unit number
    - significant_tokens: high-information locality tokens (excluding generic words)
    """
    raw_str = "" if raw_address is None else str(raw_address)
    clean_a = clean_text(raw_str)
    tokens = clean_a.split()

    raw_digits = RE_DIGITS.findall(clean_a)
    digits = [clean_digit(d) for d in raw_digits]

    first_digit = digits[0] if digits else None
    second_digit = digits[1] if len(digits) > 1 else None

    # Filter out generic street / direction words
    filtered = [
        t for t in tokens
        if len(t) >= 4 and t not in GENERIC_ADDR_TOKENS
    ]

    # Keep beginning (house/locality) and end (city/state) tokens
    if len(filtered) <= 6:
        sig_tokens = filtered
    else:
        sig_tokens = filtered[:3] + filtered[-3:]

    return {
        "raw_address": raw_str,
        "clean_address": clean_a,
        "tokens": tokens,
        "digits": digits[:3],
        "first_digit": first_digit,
        "second_digit": second_digit,
        "significant_tokens": sig_tokens,
    }


def normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize one business entity record into multiple rich representations.

    Parameters
    ----------
    record : dict
        Raw entity record containing:
        - entity_id: str
        - business_name: str
        - business_address: str
        - country: str

    Returns
    -------
    dict
        Multiple normalized representations for candidate generation and feature extraction.
    """
    entity_id = str(record.get("entity_id", ""))
    country_raw = record.get("country", "")
    country = str(country_raw).strip().upper() if country_raw else "UNKNOWN"

    name_norm = normalize_business_name(record.get("business_name", ""))
    addr_norm = normalize_address(record.get("business_address", ""))

    return {
        "entity_id": entity_id,
        "country": country,
        "business_name": name_norm["raw_name"],
        "business_address": addr_norm["raw_address"],
        "clean_name": name_norm["clean_name"],
        "core_name": name_norm["core_name"],
        "name_tokens": name_norm["tokens"],
        "core_tokens": name_norm["core_tokens"],
        "sorted_core": name_norm["sorted_core"],
        "compact_forms": name_norm["compact_forms"],
        "domain_root": name_norm["domain_root"],
        "clean_address": addr_norm["clean_address"],
        "address_tokens": addr_norm["tokens"],
        "digits": addr_norm["digits"],
        "first_digit": addr_norm["first_digit"],
        "second_digit": addr_norm["second_digit"],
        "significant_address_tokens": addr_norm["significant_tokens"],
    }
