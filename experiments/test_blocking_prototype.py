import os
import sys
import time
import re
import html
import unicodedata
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.abspath("."))

def normalize_text_quick(text):
    if not text:
        return ""
    text = html.unescape(str(text))
    text = unicodedata.normalize("NFKC", text)
    # Strip latin accents
    chars = [c for c in unicodedata.normalize("NFKD", text) if not ("\u0300" <= c <= "\u036f")]
    text = "".join(chars).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def get_tokens(text):
    return text.split()

def get_digits(text):
    return re.findall(r"\b\d+\b", text)

LEGAL_SUFFIXES = {
    "corp", "corporation", "inc", "incorporated", "co", "company",
    "ltd", "limited", "pvt", "private", "llc", "llp", "sa", "sarl", "sas", "sasu"
}

def get_core_tokens(tokens):
    return [t for t in tokens if t not in LEGAL_SUFFIXES]

# Let's inspect on a sample of ground truth
print("Loading ground truth sample...")
gt_map = {}
gt_target_s2_s3 = set()
with open("student_resource/dataset/train/train_ground_truth.tsv", encoding="utf-8") as f:
    next(f)
    for i, line in enumerate(f):
        if i >= 10000:
            break
        p = line.strip().split("\t")
        s1_id = p[0]
        matches = [m.strip() for m in p[1].split(",") if m.strip()] if len(p) > 1 and p[1] else []
        gt_map[s1_id] = set(matches)
        for m in matches:
            gt_target_s2_s3.add(m)

print(f"Loaded {len(gt_map)} S1 ground truth records with {len(gt_target_s2_s3)} true S2/S3 target IDs.")

# Load the corresponding S1 records
s1_records = {}
with open("student_resource/dataset/train/train_source1.tsv", encoding="utf-8") as f:
    next(f)
    for line in f:
        p = line.strip().split("\t")
        if p[0] in gt_map:
            s1_records[p[0]] = {
                "entity_id": p[0],
                "business_name": p[1],
                "business_address": p[2] if len(p) > 2 else "",
                "country": p[3].upper() if len(p) > 3 else "UNKNOWN"
            }
            if len(s1_records) == len(gt_map):
                break

print(f"Loaded {len(s1_records)} S1 full records.")

# We will index a pool of S2 and S3 records:
# All true targets + 200,000 random background records to test realistic noise and candidate dilution!
print("Loading candidate pool (targets + background noise)...")
candidate_pool = {}

def load_source(path, prefix, max_bg=100000):
    bg_loaded = 0
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.strip().split("\t")
            sid = p[0]
            if sid in gt_target_s2_s3:
                candidate_pool[sid] = {
                    "entity_id": sid,
                    "business_name": p[1],
                    "business_address": p[2] if len(p) > 2 else "",
                    "country": p[3].upper() if len(p) > 3 else "UNKNOWN"
                }
            elif bg_loaded < max_bg:
                candidate_pool[sid] = {
                    "entity_id": sid,
                    "business_name": p[1],
                    "business_address": p[2] if len(p) > 2 else "",
                    "country": p[3].upper() if len(p) > 3 else "UNKNOWN"
                }
                bg_loaded += 1

load_source("student_resource/dataset/train/train_source2.tsv", "S2", max_bg=100000)
load_source("student_resource/dataset/train/train_source3.tsv", "S3", max_bg=100000)

print(f"Candidate pool total size: {len(candidate_pool)} records (including {sum(1 for k in gt_target_s2_s3 if k in candidate_pool)} true targets).")

# Enhanced transliterator
DEVA_TO_LATIN = {
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

COMMON_INDIC_WORDS = {
    'प्राइवेट': 'pvt', 'लिमिटेड': 'ltd', 'प्रा': 'pvt', 'लि': 'ltd',
    'एलएलपी': 'llp', 'कंपनी': 'co', 'इन्वेस्टमेंट': 'investment',
    'डेवलपर्स': 'developers', 'फूड': 'food', 'वेंचर्स': 'ventures',
    'मार्केटिंग': 'marketing', 'प्रॉपर्टीज': 'properties',
    'कंस्ट्रक्शंस': 'constructions', 'टेक्नोलॉजीज': 'technologies',
    'एंटरप्राइजेज': 'enterprises', 'होटल': 'hotel', 'रियल': 'real',
    'मॉडर्न': 'modern', 'फर्स्ट': 'first', 'ग्लोबल': 'global',
    'अल्फा': 'alpha', 'अल': 'al', 'जैन': 'jain', 'लक्ष्मी': 'laxmi',
    'राम': 'ram', 'आदित्य': 'aditya', 'सन': 'sun', 'एग्रो': 'agro',
    'इन्फ्रा': 'infra', 'एस्टेट': 'estate', 'शक्ति': 'shakti'
}

def transliterate_indic(text):
    if not text:
        return ""
    # Map Indic Brahmic scripts to Devanagari block
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
    for term, rep in COMMON_INDIC_WORDS.items():
        deva_text = deva_text.replace(term, rep)
    res = []
    i = 0
    n = len(deva_text)
    while i < n:
        c = deva_text[i]
        if c in DEVA_TO_LATIN:
            val = DEVA_TO_LATIN[c]
            if i + 1 < n and deva_text[i+1] == '\u094d':
                res.append(val)
                i += 2
                continue
            elif i + 1 < n and deva_text[i+1] in ['ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'े', 'ै', 'ो', 'ौ', 'ं', 'ः']:
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

def clean_digit(d):
    s = d.lstrip("0")
    return s if s else "0"

LEET_MAP = str.maketrans({"@": "a", "0": "o", "1": "i", "3": "e", "$": "s"})

def clean_and_normalize(record):
    raw_name = str(record["business_name"])
    raw_addr = str(record["business_address"])
    country = str(record["country"]).upper()

    # Domain extraction from raw name (handles .com, .org, .c0m, etc.)
    norm_for_dom = raw_name.lower().translate(LEET_MAP)
    domain_root = None
    m_dom = re.search(r"([a-z0-9]+)\.(?:com|org|net|in|co|io|fr|gov|edu)", norm_for_dom)
    if m_dom:
        domain_root = m_dom.group(1)

    name = raw_name
    if name.startswith("@") or name.startswith("#"):
        name = name[1:]
    name = transliterate_indic(name)
    clean_n = normalize_text_quick(name)
    leet_n = clean_n.translate(LEET_MAP)

    tokens_n = clean_n.split()
    leet_tokens = leet_n.split()
    core_n_tokens = [t for t in tokens_n if t not in LEGAL_SUFFIXES]
    leet_core_tokens = [t for t in leet_tokens if t not in LEGAL_SUFFIXES]

    core_n = " ".join(core_n_tokens)
    compact_core = "".join(core_n_tokens)
    compact_full = "".join(tokens_n)
    compact_leet = "".join(leet_core_tokens)

    first2_compact = "".join(core_n_tokens[:2]) if len(core_n_tokens) >= 2 else None
    sorted_core = " ".join(sorted(core_n_tokens[:3])) if core_n_tokens else ""

    addr = transliterate_indic(raw_addr)
    clean_a = normalize_text_quick(addr)
    tokens_a = clean_a.split()
    raw_digits = re.findall(r"\d+", clean_a)
    digits_a = [clean_digit(d) for d in raw_digits]
    
    first_digit = digits_a[0] if digits_a else None
    second_digit = digits_a[1] if len(digits_a) > 1 else None

    filtered_addr_tokens = [
        t for t in tokens_a
        if len(t) >= 4 and t not in {
            "road", "street", "avenue", "drive", "lane", "court", "suite", "floor",
            "north", "south", "east", "west", "near", "opposite", "behind", "phase",
            "sector", "nagar", "colony", "city", "plot", "building", "state", "india"
        }
    ]
    if len(filtered_addr_tokens) <= 6:
        addr_sig_tokens = filtered_addr_tokens
    else:
        addr_sig_tokens = filtered_addr_tokens[:3] + filtered_addr_tokens[-3:]

    all_compact = {compact_core, compact_full, compact_leet}
    if domain_root:
        all_compact.add(domain_root)
    if first2_compact:
        all_compact.add(first2_compact)
    all_compact = {c for c in all_compact if len(c) >= 4}

    return {
        "entity_id": record["entity_id"],
        "country": country,
        "clean_name": clean_n,
        "core_name": core_n,
        "all_compact": all_compact,
        "sorted_core": sorted_core,
        "core_tokens": set(core_n_tokens) | set(leet_core_tokens),
        "digits_a": digits_a[:2],
        "first_digit": first_digit,
        "second_digit": second_digit,
        "addr_sig_tokens": addr_sig_tokens
    }

print("\nBuilding multi-indexes on candidate pool...")
t0 = time.time()

index_exact_core = defaultdict(list)
index_sorted_core = defaultdict(list)
index_compact = defaultdict(list)
index_token = defaultdict(list)
index_addr_num_word = defaultdict(list)
index_addr_two_num = defaultdict(list)
token_freq = defaultdict(int)

# Pre-parse candidate pool
for cid, rec in candidate_pool.items():
    p = clean_and_normalize(rec)
    c = p["country"]
    
    # 1. Exact core name
    if p["core_name"]:
        index_exact_core[(c, p["core_name"])].append(cid)
    # 2. Sorted core tokens
    if p["sorted_core"]:
        index_sorted_core[(c, p["sorted_core"])].append(cid)
    # 3. Compact / domain name
    for comp in p["all_compact"]:
        index_compact[(c, comp)].append(cid)
    # 4. Name tokens
    for t in p["core_tokens"]:
        if len(t) >= 3:
            token_freq[(c, t)] += 1
            index_token[(c, t)].append(cid)
    # 5. Address index: digit + locality word
    for d in p["digits_a"]:
        for at in p["addr_sig_tokens"]:
            index_addr_num_word[(c, d, at)].append(cid)
    # 6. Address index: two digits
    if p["first_digit"] and p["second_digit"]:
        index_addr_two_num[(c, p["first_digit"], p["second_digit"])].append(cid)

print(f"Index built in {time.time()-t0:.2f}s")

MAX_NAME_TOKEN_DOCS = 120
MAX_CANDIDATES_PER_S1 = 80

print("\nEvaluating candidate generation on S1 sample...")
t1 = time.time()

total_true_matches = sum(len(m) for m in gt_map.values())
found_true_matches = 0
candidate_counts = []

for s1_id, s1_rec in s1_records.items():
    true_matches = gt_map.get(s1_id, set())
    p1 = clean_and_normalize(s1_rec)
    c = p1["country"]
    
    # Scored candidate collection
    cand_scores = defaultdict(float)
    
    # 1. Exact core name (Highest priority)
    if p1["core_name"]:
        for cid in index_exact_core.get((c, p1["core_name"]), ()):
            cand_scores[cid] += 5.0
            
    # 2. Sorted core tokens (High priority)
    if p1["sorted_core"]:
        for cid in index_sorted_core.get((c, p1["sorted_core"]), ()):
            cand_scores[cid] += 4.0
            
    # 3. Compact / domain name (High priority)
    for comp in p1["all_compact"]:
        for cid in index_compact.get((c, comp), ()):
            cand_scores[cid] += 3.5
            
    # 4. Address: house number + locality token (Medium-High priority)
    for d in p1["digits_a"]:
        for at in p1["addr_sig_tokens"]:
            for cid in index_addr_num_word.get((c, d, at), ()):
                cand_scores[cid] += 3.0
                
    # 5. Address: two digits (Medium priority)
    if p1["first_digit"] and p1["second_digit"]:
        for cid in index_addr_two_num.get((c, p1["first_digit"], p1["second_digit"]), ()):
            cand_scores[cid] += 2.0

    # 6. Informative name tokens (rare tokens only, capped)
    for t in p1["core_tokens"]:
        if len(t) >= 4:
            cands = index_token.get((c, t), ())
            if 0 < len(cands) <= MAX_NAME_TOKEN_DOCS:
                for cid in cands:
                    cand_scores[cid] += 1.0
                    
    # Strict top-K capping sorted by match priority score
    MAX_CANDIDATES = 60
    if len(cand_scores) > MAX_CANDIDATES:
        sorted_cands = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)
        candidates = {x[0] for x in sorted_cands[:MAX_CANDIDATES]}
    else:
        candidates = set(cand_scores.keys())

    candidate_counts.append(len(candidates))
    
    for tm in true_matches:
        if tm in candidates:
            found_true_matches += 1

elapsed = time.time() - t1
candidate_counts.sort()
import numpy as np

arr = np.array(candidate_counts)
recall = found_true_matches / total_true_matches if total_true_matches > 0 else 0.0

print("="*60)
print(f"PRODUCTION-GRADE BLOCKING BENCHMARK (10,000 S1 records):")
print(f"Total True Matches: {total_true_matches}")
print(f"Found True Matches: {found_true_matches}")
print(f"True-Match Recall:  {recall:.4%} ({found_true_matches}/{total_true_matches})")
print(f"Average candidates / S1: {arr.mean():.2f}")
print(f"Median candidates / S1:  {np.median(arr):.1f}")
print(f"P90 candidates / S1:     {np.percentile(arr, 90):.1f}")
print(f"P95 candidates / S1:     {np.percentile(arr, 95):.1f}")
print(f"P99 candidates / S1:     {np.percentile(arr, 99):.1f}")
print(f"Max candidates / S1:     {arr.max()}")
print(f"Total S1 query time:     {elapsed:.2f}s ({len(s1_records)/elapsed:.1f} S1/sec)")
print("="*60)


