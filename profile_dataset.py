import pandas as pd
import os

BASE = r"C:\Users\SAURABH LUBAL\Downloads\amazon_ml\student_resource\dataset"
TRAIN = os.path.join(BASE, "train")

files = {
    "Source 1": os.path.join(TRAIN, "train_source1.tsv"),
    "Source 2": os.path.join(TRAIN, "train_source2.tsv"),
    "Source 3": os.path.join(TRAIN, "train_source3.tsv"),
    "Ground Truth": os.path.join(TRAIN, "train_ground_truth.tsv"),
}

report = []

def add(text=""):
    print(text)
    report.append(text)


# ============================================================
# PROFILE SOURCE FILES
# ============================================================

for name in ["Source 1", "Source 2", "Source 3"]:

    path = files[name]

    add("\n" + "=" * 70)
    add(name.upper())
    add("=" * 70)

    # Read in chunks so we don't consume huge amounts of RAM
    total_rows = 0
    missing = None
    countries = {}
    duplicate_ids = 0
    first_rows = None

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=100_000
    ):

        total_rows += len(chunk)

        if first_rows is None:
            first_rows = chunk.head(3).copy()
            columns = list(chunk.columns)
            missing = {col: 0 for col in columns}

        # Missing values
        for col in columns:
            missing[col] += (chunk[col].str.strip() == "").sum()

        # Country counts
        if "country" in chunk.columns:
            counts = chunk["country"].value_counts()

            for country, count in counts.items():
                countries[country] = countries.get(country, 0) + count

        # Duplicate entity IDs within the file
        if "entity_id" in chunk.columns:
            duplicate_ids += chunk["entity_id"].duplicated().sum()

    add(f"Rows: {total_rows:,}")
    add(f"Columns: {columns}")

    add("\nMissing values:")

    for col, count in missing.items():
        percentage = count / total_rows * 100
        add(f"  {col}: {count:,} ({percentage:.2f}%)")

    add("\nCountries:")

    for country, count in sorted(
        countries.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        percentage = count / total_rows * 100
        add(f"  {country}: {count:,} ({percentage:.2f}%)")

    add(f"\nDuplicate entity IDs: {duplicate_ids:,}")

    add("\nFirst 3 records:")

    for _, row in first_rows.iterrows():
        add(str(row.to_dict()))


# ============================================================
# PROFILE GROUND TRUTH
# ============================================================

path = files["Ground Truth"]

add("\n" + "=" * 70)
add("GROUND TRUTH")
add("=" * 70)

gt = pd.read_csv(
    path,
    sep="\t",
    dtype=str,
    keep_default_na=False
)

add(f"Rows: {len(gt):,}")
add(f"Columns: {list(gt.columns)}")


# Number of matched entities per S1
match_counts = gt["matched_entity_ids"].apply(
    lambda x: 0 if not x.strip() else len(x.split(","))
)

add("\nMatches per Source-1 entity:")

add(f"  Minimum: {match_counts.min()}")
add(f"  Maximum: {match_counts.max()}")
add(f"  Mean: {match_counts.mean():.2f}")
add(f"  Median: {match_counts.median():.0f}")


# Distribution
add("\nMatch-count distribution:")

distribution = match_counts.value_counts().sort_index()

for count, frequency in distribution.items():
    add(f"  {count} matches: {frequency:,}")


# S2 and S3 matches
s2_counts = gt["matched_entity_ids"].apply(
    lambda x: sum(
        1 for item in x.split(",")
        if item.startswith("S2-")
    )
)

s3_counts = gt["matched_entity_ids"].apply(
    lambda x: sum(
        1 for item in x.split(",")
        if item.startswith("S3-")
    )
)

add("\nSource-specific matches:")

add(f"  Total S2 matches: {s2_counts.sum():,}")
add(f"  Total S3 matches: {s3_counts.sum():,}")

add(
    f"  Average S2 matches per S1: "
    f"{s2_counts.mean():.2f}"
)

add(
    f"  Average S3 matches per S1: "
    f"{s3_counts.mean():.2f}"
)


# First examples
add("\nFirst 10 ground-truth records:")

for _, row in gt.head(10).iterrows():

    add(
        f"{row['source1_entity_id']} -> "
        f"{row['matched_entity_ids']}"
    )


# ============================================================
# SAVE REPORT
# ============================================================

output = "dataset_profile.txt"

with open(output, "w", encoding="utf-8") as f:
    f.write("\n".join(report))

print("\n" + "=" * 70)
print("PROFILE COMPLETE")
print(f"Report saved to: {os.path.abspath(output)}")
print("=" * 70)