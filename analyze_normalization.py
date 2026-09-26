import pandas as pd
import re
import unicodedata
import html
import os


BASE = r"C:\Users\SAURABH LUBAL\Downloads\amazon_ml\student_resource\dataset"
TRAIN = os.path.join(BASE, "train")


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    if not text:
        return ""

    text = str(text)

    # HTML entities
    text = html.unescape(text)

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Lowercase
    text = text.lower()

    # Replace punctuation with spaces
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_name(text):

    text = normalize_text(text)

    # Common business suffixes
    replacements = {
        "corporation": "corp",
        "incorporated": "inc",
        "company": "co",
        "limited": "ltd",
        "private": "pvt",
        "privatelimited": "pvt ltd",
    }

    words = text.split()

    words = [
        replacements.get(word, word)
        for word in words
    ]

    return " ".join(words)


def normalize_address(text):

    text = normalize_text(text)

    replacements = {
        "road": "rd",
        "street": "st",
        "avenue": "ave",
        "boulevard": "blvd",
        "highway": "hwy",
        "drive": "dr",
        "lane": "ln",
        "court": "ct",
        "parkway": "pkwy",
        "apartment": "apt",
        "suite": "ste",
    }

    words = text.split()

    words = [
        replacements.get(word, word)
        for word in words
    ]

    return " ".join(words)


# ============================================================
# LOAD SAMPLE DATA
# ============================================================

files = {
    "S1": os.path.join(TRAIN, "train_source1.tsv"),
    "S2": os.path.join(TRAIN, "train_source2.tsv"),
    "S3": os.path.join(TRAIN, "train_source3.tsv"),
}


print("=" * 70)
print("NORMALIZATION EXPERIMENT")
print("=" * 70)


# Read only a sample first
s1 = pd.read_csv(
    files["S1"],
    sep="\t",
    dtype=str,
    keep_default_na=False,
    nrows=10000
)

s2 = pd.read_csv(
    files["S2"],
    sep="\t",
    dtype=str,
    keep_default_na=False,
    nrows=10000
)

s3 = pd.read_csv(
    files["S3"],
    sep="\t",
    dtype=str,
    keep_default_na=False,
    nrows=10000
)


# ============================================================
# SHOW EXAMPLES
# ============================================================

print("\nNAME NORMALIZATION EXAMPLES")
print("-" * 70)

for name in s1["business_name"].head(20):

    print(f"Original : {name}")
    print(f"Normalized: {normalize_name(name)}")
    print()


print("\nADDRESS NORMALIZATION EXAMPLES")
print("-" * 70)

for address in s1["business_address"].head(20):

    print(f"Original : {address}")
    print(f"Normalized: {normalize_address(address)}")
    print()


# ============================================================
# COUNTRY DISTRIBUTION
# ============================================================

print("\nCOUNTRIES IN TRAINING SAMPLE")
print("-" * 70)

for source, df in [
    ("S1", s1),
    ("S2", s2),
    ("S3", s3)
]:

    print(source)

    for country, count in df["country"].value_counts().items():
        print(f"  {country}: {count}")


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)