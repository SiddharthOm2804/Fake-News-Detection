"""WELFake Dataset Preprocessing and Standardization Pipeline.

STEP 16: WELFake Dataset Integration.
- Locates and inspects WELFake_Dataset.csv
- Normalizes label convention (WELFake 0 -> 1 REAL, WELFake 1 -> 0 FAKE)
- Combines title + text into a unified 'text' column
- Applies project standard NLP preprocessing (preprocess_text)
- Removes duplicates, NaNs, and empty records
- Exports standardized dataset to data/processed/welfake_cleaned_news.csv
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np

from src.data_preprocessing import preprocess_text


def find_welfake_csv(dataset_dir: Path | None = None) -> Path:
    """Locate WELFake CSV inside the dataset directory.

    Args:
        dataset_dir: Directory containing dataset files.

    Returns:
        Path: Path to the discovered WELFake CSV file.

    Raises:
        FileNotFoundError: If no WELFake file is found.
    """
    if dataset_dir is None:
        project_root = Path(__file__).resolve().parent.parent
        dataset_dir = project_root / "dataset"

    # 1. Look for standard expected name
    standard_path = dataset_dir / "WELFake_Dataset.csv"
    if standard_path.exists():
        return standard_path

    # 2. Case-insensitive or wildcard search
    csv_candidates = list(dataset_dir.glob("*.csv"))
    for candidate in csv_candidates:
        if "welfake" in candidate.name.lower():
            return candidate

    raise FileNotFoundError(
        f"Error: WELFake CSV file could not be found inside '{dataset_dir}'. "
        f"Existing files: {[f.name for f in csv_candidates]}"
    )


def inspect_welfake_dataset(file_path: Path) -> pd.DataFrame:
    """Perform initial inspection of raw WELFake dataset.

    Args:
        file_path: Path to raw WELFake CSV.

    Returns:
        pd.DataFrame: Raw loaded DataFrame.
    """
    print("=" * 70)
    print("        WELFake DATASET: INITIAL STRUCTURAL INSPECTION")
    print("=" * 70)
    print(f"[INFO] Loading raw WELFake file: {file_path}")

    df = pd.read_csv(file_path)
    total_raw_rows = len(df)

    print("\n--- 1. Schema & Dimensions ---")
    print(f"Total Rows: {total_raw_rows:,}")
    print(f"Total Columns: {len(df.columns)}")
    print(f"Column Names: {list(df.columns)}")

    print("\n--- 2. Data Types ---")
    for col, dt in df.dtypes.items():
        print(f"  - {col:<15} ({dt})")

    print("\n--- 3. Missing / Null Values ---")
    for col, null_count in df.isnull().sum().items():
        pct = (null_count / total_raw_rows) * 100
        print(f"  - {col:<15}: {null_count:,} nulls ({pct:.2f}%)")

    print("\n--- 4. Raw Duplicates Check ---")
    raw_dups = df.duplicated().sum()
    print(f"Exact Duplicate Rows across all columns: {raw_dups:,}")

    print("\n--- 5. Raw Label Distribution (Original WELFake convention) ---")
    print("Note: In raw WELFake: 0 = REAL, 1 = FAKE")
    print(df["label"].value_counts(dropna=False).to_string())

    return df


def process_welfake_dataset(
    input_path: Path | None = None,
    output_path: Path | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Standardize, clean, label-convert, and save WELFake dataset.

    Pipeline:
    1. Identify columns ('title', 'text', 'label')
    2. Convert labels to project convention:
       - WELFake 0 (REAL) -> project 1 (REAL)
       - WELFake 1 (FAKE) -> project 0 (FAKE)
    3. Safely combine title + text (without inserting 'nan' strings)
    4. Apply NLP cleaning (preprocess_text)
    5. Deduplicate and remove empty records
    6. Shuffle and reset index
    7. Save to data/processed/welfake_cleaned_news.csv

    Args:
        input_path: Path to raw WELFake CSV.
        output_path: Destination path for standardized CSV.
        random_state: Seed for shuffling.

    Returns:
        pd.DataFrame: Cleaned and standardized DataFrame.
    """
    project_root = Path(__file__).resolve().parent.parent

    if input_path is None:
        input_path = find_welfake_csv(project_root / "dataset")

    if output_path is None:
        output_path = project_root / "data" / "processed" / "welfake_cleaned_news.csv"

    # Step 2: Inspect raw dataset
    raw_df = inspect_welfake_dataset(input_path)
    initial_row_count = len(raw_df)

    print("\n" + "=" * 70)
    print("        STANDARDIZING & PREPROCESSING WELFake DATASET")
    print("=" * 70)

    # 3. Verify and identify columns
    cols_lower = {str(c).lower().strip(): c for c in raw_df.columns}

    if "label" not in cols_lower:
        raise KeyError("Required column 'label' not found in WELFake dataset.")
    if "text" not in cols_lower:
        raise KeyError("Required column 'text' not found in WELFake dataset.")

    actual_label_col = cols_lower["label"]
    actual_text_col = cols_lower["text"]
    actual_title_col = cols_lower.get("title", None)

    print(f"[INFO] Detected Columns -> Title: '{actual_title_col}', Text: '{actual_text_col}', Label: '{actual_label_col}'")

    # Filter out any non-integer or NaN labels
    df_clean = raw_df.dropna(subset=[actual_label_col]).copy()
    df_clean = df_clean[df_clean[actual_label_col].isin([0, 1, 0.0, 1.0])].copy()
    df_clean[actual_label_col] = df_clean[actual_label_col].astype(int)

    # Keep original label for verification reporting BEFORE mapping
    df_clean["orig_label"] = df_clean[actual_label_col].copy()

    # 3. NORMALIZE LABELS TO PROJECT CONVENTION:
    # WELFake 0 (REAL) -> Project 1 (REAL)
    # WELFake 1 (FAKE) -> Project 0 (FAKE)
    print("\n[INFO] Normalizing label convention to project standard:")
    print("  - WELFake Label 0 (REAL) -> Project Label 1 (REAL)")
    print("  - WELFake Label 1 (FAKE) -> Project Label 0 (FAKE)")
    df_clean["label"] = df_clean[actual_label_col].map({0: 1, 1: 0})

    # 4. Safely combine Title + Text into unified 'text' column
    print("\n[INFO] Combining 'title' + 'text' into single unified 'text' column...")
    if actual_title_col and actual_title_col in df_clean.columns:
        title_series = df_clean[actual_title_col].fillna("").astype(str).str.strip()
    else:
        title_series = pd.Series([""] * len(df_clean), index=df_clean.index)

    text_series = df_clean[actual_text_col].fillna("").astype(str).str.strip()

    # Combine: title + " " + text without adding 'nan'
    df_clean["text"] = (title_series + " " + text_series).str.strip()

    # Retain standardized schema with orig_label for inspection
    df_clean = df_clean[["text", "label", "orig_label"]].copy()

    # 6. Remove completely empty text records before NLP cleaning
    df_clean = df_clean[df_clean["text"].str.strip() != ""].copy()
    empty_initially_removed = initial_row_count - len(df_clean)

    # 6. Remove duplicates on combined raw text
    dups_raw_count = int(df_clean.duplicated(subset=["text"]).sum())
    df_clean = df_clean.drop_duplicates(subset=["text"]).copy()

    print(f"Raw duplicates removed: {dups_raw_count:,}")

    # 5. Apply project NLP text cleaning pipeline
    print(f"\n[INFO] Applying NLP text cleaning (preprocess_text) to {len(df_clean):,} records...")
    df_clean["text"] = df_clean["text"].apply(preprocess_text)

    # Remove records that became empty after NLP preprocessing
    pre_nlp_count = len(df_clean)
    df_clean = df_clean[df_clean["text"].str.strip() != ""].copy()
    empty_post_nlp = pre_nlp_count - len(df_clean)
    total_empty_removed = empty_initially_removed + empty_post_nlp

    # Remove any duplicates created post-NLP cleaning
    dups_post_nlp = int(df_clean.duplicated(subset=["text"]).sum())
    df_clean = df_clean.drop_duplicates(subset=["text"]).copy()
    total_duplicates_removed = dups_raw_count + dups_post_nlp

    # 6. Shuffle dataset reproducibly
    print(f"\n[INFO] Shuffling dataset with random_state={random_state}...")
    df_clean = df_clean.sample(frac=1.0, random_state=random_state)
    df_clean.reset_index(drop=True, inplace=True)

    # Final counts and metrics
    final_total = len(df_clean)
    total_removed = initial_row_count - final_total
    real_count = int((df_clean["label"] == 1).sum())
    fake_count = int((df_clean["label"] == 0).sum())

    # 8. Print clear dataset report
    print("\n" + "=" * 70)
    print("             WELFake DATASET PROCESSING REPORT")
    print("=" * 70)
    print(f"Total articles (Cleaned)           : {final_total:,}")
    print(f"REAL articles (Label 1)            : {real_count:,} ({real_count / final_total * 100:.2f}%)")
    print(f"FAKE articles (Label 0)            : {fake_count:,} ({fake_count / final_total * 100:.2f}%)")
    print(f"Duplicates removed                 : {total_duplicates_removed:,}")
    print(f"Missing/empty records removed      : {total_empty_removed:,}")
    print(f"Total records removed in cleaning  : {total_removed:,}")

    # 9. Verify final labels
    unique_labels = set(df_clean["label"].unique())
    if unique_labels != {0, 1}:
        raise ValueError(f"Validation Error: Expected labels {{0, 1}}, got: {unique_labels}")
    print("\n[VALIDATION PASSED] Labels verified: 0 = FAKE, 1 = REAL strictly.")

    # 10. Sanity check: Verify several records manually
    print("\n" + "=" * 70)
    print("        VERIFYING SAMPLE RECORDS (ORIGINAL vs CONVERTED)")
    print("=" * 70)

    print("\n>>> SAMPLE FAKE RECORDS (WELFake 1 -> Project 0) <<<\n")
    fake_samples = df_clean[df_clean["label"] == 0].head(3)
    for idx, (_, row) in enumerate(fake_samples.iterrows(), start=1):
        sample_snippet = row["text"][:180] + "..." if len(row["text"]) > 180 else row["text"]
        print(f"Record #{idx}:")
        print(f"  Original WELFake Label : {row['orig_label']}")
        print(f"  Converted Project Label: {row['label']}")
        print(f"  Text preview           : \"{sample_snippet}\"\n")

    print(">>> SAMPLE REAL RECORDS (WELFake 0 -> Project 1) <<<\n")
    real_samples = df_clean[df_clean["label"] == 1].head(3)
    for idx, (_, row) in enumerate(real_samples.iterrows(), start=1):
        sample_snippet = row["text"][:180] + "..." if len(row["text"]) > 180 else row["text"]
        print(f"Record #{idx}:")
        print(f"  Original WELFake Label : {row['orig_label']}")
        print(f"  Converted Project Label: {row['label']}")
        print(f"  Text preview           : \"{sample_snippet}\"\n")

    # 7. Save strictly ['text', 'label'] to data/processed/welfake_cleaned_news.csv
    df_clean_to_save = df_clean[["text", "label"]].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Saving cleaned dataset to: {output_path}")
    df_clean_to_save.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[SUCCESS] Saved {len(df_clean_to_save):,} records to '{output_path.name}'.")

    return df_clean_to_save


def main():
    """Main CLI execution entry point."""
    process_welfake_dataset()


if __name__ == "__main__":
    main()
