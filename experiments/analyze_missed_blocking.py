import os
import sys
import re
import html
import unicodedata
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.abspath("."))
sys.stdout.reconfigure(encoding="utf-8")

def normalize_text_quick(text):
    if not text:
        return ""
    text = html.unescape(str(text))
    text = unicodedata.normalize("NFKC", text)
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

HINDI_TERMS = {
    'प्राइवेट': 'pvt', 'लिमिटेड': 'ltd', 'प्रा': 'pvt', 'लि': 'ltd',
    'एलएलपी': 'llp', 'कंपनी': 'co', 'इन्वेस्टमेंट': 'investment',
    'डेवलपर्स': 'developers', 'फूड': 'food', 'वेंचर्स': 'ventures',
    'मार्केटिंग': 'marketing', 'प्रॉपर्टीज': 'properties',
    'कंस्ट्रक्शंस': 'constructions', 'टेक्नोलॉजीज': 'technologies',
    'एंटरप्राइजेज': 'enterprises', 'होटल': 'hotel', 'रियल': 'real',
    'मॉडर्न': 'modern', 'फर्स्ट': 'first', 'ग्लोबल': 'global',
    'अल्फा': 'alpha', 'अल': 'al', 'जैन': 'jain', 'लक्ष्मी': 'laxmi',
    'राम': 'ram', 'आदित्य': 'aditya', 'सन': 'sun'
}

def clean_and_normalize(record):
    raw_name = record["business_name"]
    raw_addr = record["business_address"]
    country = record["country"]

    name = raw_name
    for hi, en in HINDI_TERMS.items():
        if hi in name:
            name = name.replace(hi, en)
    
    clean_n = normalize_text_quick(name)
    tokens_n = get_tokens(clean_n)
    core_n_tokens = get_core_tokens(tokens_n)
    core_n = " ".join(core_n_tokens)

    compact_n = "".join(tokens_n)
    domain_root = None
    m = re.search(r"([a-z0-9]+)\.(?:com|org|net|in|co|io|fr)", clean_n)
    if m:
        domain_root = m.group(1)

    sorted_core = " ".join(sorted(core_n_tokens[:4])) if core_n_tokens else ""

    clean_a = normalize_text_quick(raw_addr)
    tokens_a = get_tokens(clean_a)
    digits_a = get_digits(clean_a)
    first_digit = digits_a[0] if digits_a else None
    addr_sig_tokens = [t for t in tokens_a if len(t) >= 4 and t not in {"road", "street", "avenue", "drive", "lane", "court", "suite", "floor", "north", "south", "east", "west"}]

    return {
        "entity_id": record["entity_id"],
        "raw_name": raw_name,
        "raw_addr": raw_addr,
        "country": country,
        "clean_name": clean_n,
        "core_name": core_n,
        "compact_name": compact_n,
        "domain_root": domain_root,
        "sorted_core": sorted_core,
        "core_tokens": set(core_n_tokens),
        "first_digit": first_digit,
        "addr_sig_tokens": addr_sig_tokens[:4]
    }

# Load ground truth for 2000 S1 records
gt_map = {}
gt_target_s2_s3 = set()
with open("student_resource/dataset/train/train_ground_truth.tsv", encoding="utf-8") as f:
    next(f)
    for i, line in enumerate(f):
        if i >= 2000:
            break
        p = line.strip().split("\t")
        s1_id = p[0]
        matches = [m.strip() for m in p[1].split(",") if m.strip()] if len(p) > 1 and p[1] else []
        gt_map[s1_id] = set(matches)
        for m in matches:
            gt_target_s2_s3.add(m)

# Load S1 records
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

# Load only the true target records to analyze missed ones
candidate_pool = {}
for path, prefix in [("student_resource/dataset/train/train_source2.tsv", "S2"), ("student_resource/dataset/train/train_source3.tsv", "S3")]:
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

index_exact_core = defaultdict(list)
index_sorted_core = defaultdict(list)
index_compact = defaultdict(list)
index_token = defaultdict(list)
index_addr = defaultdict(list)

for cid, rec in candidate_pool.items():
    p = clean_and_normalize(rec)
    c = p["country"]
    if p["core_name"]:
        index_exact_core[(c, p["core_name"])].append(cid)
    if p["sorted_core"]:
        index_sorted_core[(c, p["sorted_core"])].append(cid)
    if p["domain_root"]:
        index_compact[(c, p["domain_root"])].append(cid)
    elif len(p["compact_name"]) >= 4:
        index_compact[(c, p["compact_name"])].append(cid)
    for t in p["core_tokens"]:
        if len(t) >= 3:
            index_token[(c, t)].append(cid)
    if p["first_digit"]:
        for at in p["addr_sig_tokens"]:
            index_addr[(c, p["first_digit"], at)].append(cid)

missed = []
for s1_id, s1_rec in s1_records.items():
    true_matches = gt_map.get(s1_id, set())
    p1 = clean_and_normalize(s1_rec)
    c = p1["country"]
    candidates = set()
    if p1["core_name"]:
        candidates.update(index_exact_core.get((c, p1["core_name"]), ()))
    if p1["sorted_core"]:
        candidates.update(index_sorted_core.get((c, p1["sorted_core"]), ()))
    if p1["domain_root"]:
        candidates.update(index_compact.get((c, p1["domain_root"]), ()))
    elif len(p1["compact_name"]) >= 4:
        candidates.update(index_compact.get((c, p1["compact_name"]), ()))
    for t in p1["core_tokens"]:
        if len(t) >= 4:
            candidates.update(index_token.get((c, t), ()))
    if p1["first_digit"]:
        for at in p1["addr_sig_tokens"]:
            candidates.update(index_addr.get((c, p1["first_digit"], at), ()))
            
    for tm in true_matches:
        if tm not in candidates and tm in candidate_pool:
            missed.append((s1_rec, candidate_pool[tm]))

print(f"Total missed matches in target-only index: {len(missed)}")
print("\nFirst 10 Missed Examples:")
for s1_rec, cand_rec in missed[:10]:
    print("="*70)
    print(f"S1:   [{s1_rec['entity_id']}] Country: {s1_rec['country']}")
    print(f"      Name: '{s1_rec['business_name']}'")
    print(f"      Addr: '{s1_rec['business_address']}'")
    print(f"CAND: [{cand_rec['entity_id']}] Country: {cand_rec['country']}")
    print(f"      Name: '{cand_rec['business_name']}'")
    print(f"      Addr: '{cand_rec['business_address']}'")
