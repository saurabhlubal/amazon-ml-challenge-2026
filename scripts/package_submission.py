"""
Packaging Script for Amazon ML Challenge 2026 Submission.
Assembles the official submission zip conforming strictly to student_resource/README.md:

<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
"""

import os
import sys
import shutil
import zipfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def create_submission_package(
    team_name: str = "antigravity_ber",
    output_dir: str = "output",
    zip_output_dir: str = "submission",
):
    print("=" * 75)
    print(f"PACKAGING FINAL SUBMISSION: {team_name}_submission.zip")
    print("=" * 75)

    matching_tsv = os.path.join(PROJECT_ROOT, output_dir, "matching_results.tsv")
    candidate_tsv = os.path.join(PROJECT_ROOT, output_dir, "candidate_pairs.tsv")

    if not os.path.isfile(matching_tsv):
        raise FileNotFoundError(f"Missing {matching_tsv}")
    if not os.path.isfile(candidate_tsv):
        raise FileNotFoundError(f"Missing {candidate_tsv}")

    os.makedirs(os.path.join(PROJECT_ROOT, zip_output_dir), exist_ok=True)
    zip_path = os.path.join(PROJECT_ROOT, zip_output_dir, f"{team_name}_submission.zip")

    # Build zip archive directly to avoid staging 6.08 GB of TSVs
    print(f"Compressing into {zip_path} (Zip64 enabled)...")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        # 1. Add output TSVs directly
        print("1. Adding output/matching_results.tsv...")
        zf.write(matching_tsv, "output/matching_results.tsv")
        print("2. Adding output/candidate_pairs.tsv...")
        zf.write(candidate_tsv, "output/candidate_pairs.tsv")

        # 2. Add source code
        print("3. Adding code/business_entity_resolution/src/...")
        src_dir = os.path.join(PROJECT_ROOT, "business_entity_resolution", "src")
        for fname in os.listdir(src_dir):
            if fname.endswith((".py", ".json")) and not fname.startswith("."):
                zf.write(os.path.join(src_dir, fname), f"code/business_entity_resolution/src/{fname}")

        # 3. Add README and requirements
        print("4. Adding code README.md and requirements.txt...")
        zf.write(os.path.join(PROJECT_ROOT, "README.md"), "code/business_entity_resolution/README.md")
        zf.write(os.path.join(PROJECT_ROOT, "requirements.txt"), "code/business_entity_resolution/requirements.txt")

        # 4. Add Documentation_template.md
        print("5. Adding Documentation_template.md...")
        doc_src = os.path.join(PROJECT_ROOT, "DOCUMENTATION.md")
        zf.write(doc_src, "Documentation_template.md")

    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"\nFinal Package Ready: {zip_path} ({zip_size_mb:.2f} MB)")
    print("Archive Contents:")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            print(f"  - {info.filename:<45} ({info.file_size:,} bytes)")
    print("=" * 75)


if __name__ == "__main__":
    create_submission_package()
