import os
import sys
import re
import html
import unicodedata
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")

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

def normalize_text_quick(text):
    if not text:
        return ""
    text = html.unescape(str(text))
    text = unicodedata.normalize("NFKC", text)
    chars = [c for c in unicodedata.normalize("NFKD", text) if not ("\u0300" <= c <= "\u036f")]
    text = "".join(chars).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

LEGAL_SUFFIXES = {
    "corp", "corporation", "inc", "incorporated", "co", "company",
    "ltd", "limited", "pvt", "private", "llc", "llp", "sa", "sarl", "sas", "sasu"
}

def clean_digit(d):
    s = d.lstrip("0")
    return s if s else "0"

LEET_MAP = str.maketrans({"@": "a", "0": "o", "1": "i", "3": "e", "$": "s"})

def clean_and_normalize(record):
    raw_name = record["business_name"]
    raw_addr = record["business_address"]
    country = record["country"]

    # Domain extraction from raw name
    domain_root = None
    m_dom = re.search(r"([a-z0-9]+)\.(?:com|org|net|in|co|io|fr|gov|edu)", raw_name.lower())
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
        "raw_name": raw_name,
        "raw_addr": raw_addr,
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

# Load 1000 S1
gt_map = {}
gt_target = set()
with open("student_resource/dataset/train/train_ground_truth.tsv", encoding="utf-8") as f:
    next(f)
    for i, line in enumerate(f):
        if i >= 1000:
            break
        p = line.strip().split("\t")
        matches = [m.strip() for m in p[1].split(",") if m.strip()] if len(p) > 1 and p[1] else []
        gt_map[p[0]] = set(matches)
        for m in matches:
            gt_target.add(m)

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

candidate_pool = {}
for path in ["student_resource/dataset/train/train_source2.tsv", "student_resource/dataset/train/train_source3.tsv"]:
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.strip().split("\t")
            if p[0] in gt_target:
                candidate_pool[p[0]] = {
                    "entity_id": p[0],
                    "business_name": p[1],
                    "business_address": p[2] if len(p) > 2 else "",
                    "country": p[3].upper() if len(p) > 3 else "UNKNOWN"
                }

index_exact_core = defaultdict(list)
index_sorted_core = defaultdict(list)
index_compact = defaultdict(list)
index_token = defaultdict(list)
index_addr_num_word = defaultdict(list)
index_addr_two_num = defaultdict(list)

for cid, rec in candidate_pool.items():
    p = clean_and_normalize(rec)
    c = p["country"]
    if p["core_name"]:
        index_exact_core[(c, p["core_name"])].append(cid)
    if p["sorted_core"]:
        index_sorted_core[(c, p["sorted_core"])].append(cid)
    for comp in p["all_compact"]:
        index_compact[(c, comp)].append(cid)
    for t in p["core_tokens"]:
        if len(t) >= 3:
            index_token[(c, t)].append(cid)
    for d in p["digits_a"]:
        for at in p["addr_sig_tokens"]:
            index_addr_num_word[(c, d, at)].append(cid)
    if p["first_digit"] and p["second_digit"]:
        index_addr_two_num[(c, p["first_digit"], p["second_digit"])].append(cid)

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
    for comp in p1["all_compact"]:
        candidates.update(index_compact.get((c, comp), ()))
    for t in p1["core_tokens"]:
        if len(t) >= 3:
            candidates.update(index_token.get((c, t), ()))
    for d in p1["digits_a"]:
        for at in p1["addr_sig_tokens"]:
            candidates.update(index_addr_num_word.get((c, d, at), ()))
    if p1["first_digit"] and p1["second_digit"]:
        candidates.update(index_addr_two_num.get((c, p1["first_digit"], p1["second_digit"]), ()))

    for tm in true_matches:
        if tm not in candidates and tm in candidate_pool:
            missed.append((s1_rec, candidate_pool[tm]))

print(f"Total true targets: {len(gt_target)}, Missed in target index: {len(missed)} ({len(missed)/len(gt_target):.2%})")
print("\nMissed Examples:")
for s1_rec, cand_rec in missed[:15]:
    print("="*70)
    print(f"S1:   [{s1_rec['entity_id']}] Country: {s1_rec['country']}")
    print(f"      Name: '{s1_rec['business_name']}'")
    print(f"      Addr: '{s1_rec['business_address']}'")
    print(f"CAND: [{cand_rec['entity_id']}] Country: {cand_rec['country']}")
    print(f"      Name: '{cand_rec['business_name']}'")
    print(f"      Addr: '{cand_rec['business_address']}'")
