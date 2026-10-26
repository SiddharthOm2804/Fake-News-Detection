"""Data preprocessing, NLP pipeline, and dataset splitting for Fake News Detection.

STEP 5: Stratified Train / Validation / Test Splitting.
Splits preprocessed dataset into:
- 64% Training
- 16% Validation
- 20% Testing
Includes strict data leakage prevention and distribution validations.
"""

from pathlib import Path
import re
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# Precompiled regular expressions for optimal NLP cleaning performance
HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
SPECIAL_CHAR_RE = re.compile(r"[^a-zA-Z0-9\s]")
MULTI_SPACE_RE = re.compile(r"\s+")


def load_dataset_file(file_path: Path, dataset_name: str) -> pd.DataFrame:
    """Load and validate a CSV dataset file with proper error handling.

    Args:
        file_path: Path to the CSV file.
        dataset_name: Human-readable name for logging.

    Returns:
        pd.DataFrame: Loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or cannot be parsed.
    """
    print(f"[INFO] Loading {dataset_name} from: {file_path}")

    if not file_path.exists():
        raise FileNotFoundError(
            f"Error: {dataset_name} file not found at '{file_path}'."
        )

    if file_path.stat().st_size == 0:
        raise ValueError(
            f"Error: {dataset_name} file at '{file_path}' is empty (0 bytes)."
        )

    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Error: {dataset_name} contains no data or headers.")
    except pd.errors.ParserError as pe:
        raise ValueError(f"Error parsing {dataset_name} CSV: {pe}")
    except Exception as e:
        raise ValueError(f"Unexpected error loading {dataset_name}: {e}")

    if df.empty:
        raise ValueError(f"Error: {dataset_name} DataFrame is empty after reading.")

    return df


def load_datasets(data_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Fake.csv and True.csv from dataset directory.

    Args:
        data_dir: Optional directory path. Defaults to 'dataset/' relative to project root.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (fake_df, true_df)
    """
    if data_dir is None:
        project_root = Path(__file__).resolve().parent.parent
        data_dir = project_root / "dataset"

    fake_path = data_dir / "Fake.csv"
    true_path = data_dir / "True.csv"

    fake_df = load_dataset_file(fake_path, "Fake.csv")
    true_df = load_dataset_file(true_path, "True.csv")

    return fake_df, true_df


def create_labels(
    fake_df: pd.DataFrame, true_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add binary target label column to Fake (0) and True (1) DataFrames.

    Mapping:
        0 = FAKE
        1 = REAL

    Args:
        fake_df: DataFrame containing fake news records.
        true_df: DataFrame containing real news records.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (fake_df, true_df) with 'label' column.
    """
    fake_df = fake_df.copy()
    true_df = true_df.copy()

    fake_df["label"] = 0
    true_df["label"] = 1

    return fake_df, true_df


def combine_title_and_text(df: pd.DataFrame) -> pd.DataFrame:
    """Combine title and text into a single 'text' column without losing title info.

    Format:
        title + " " + article_text

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with combined 'text' column.

    Raises:
        KeyError: If the 'text' column does not exist.
    """
    df = df.copy()

    if "text" not in df.columns:
        raise KeyError(
            "Column 'text' does not exist in dataset. Cannot construct training text."
        )

    text_col = df["text"].fillna("").astype(str)

    if "title" in df.columns:
        title_col = df["title"].fillna("").astype(str)
        df["text"] = (title_col + " " + text_col).str.strip()
    else:
        df["text"] = text_col.str.strip()

    return df


def clean_dataset(
    fake_df: pd.DataFrame, true_df: pd.DataFrame, random_state: int = 42
) -> pd.DataFrame:
    """Clean, combine, deduplicate, shuffle, and validate the initial dataset.

    Args:
        fake_df: Fake news DataFrame with 'label' column.
        true_df: Real news DataFrame with 'label' column.
        random_state: Seed for reproducible shuffling.

    Returns:
        pd.DataFrame: Cleaned and combined DataFrame with ['text', 'label'].
    """
    print("\n" + "=" * 50)
    print("COMBINING & CLEANING DATASETS")
    print("=" * 50)

    fake_df_proc = combine_title_and_text(fake_df)
    true_df_proc = combine_title_and_text(true_df)

    fake_df_proc = fake_df_proc[["text", "label"]]
    true_df_proc = true_df_proc[["text", "label"]]

    combined = pd.concat([fake_df_proc, true_df_proc], ignore_index=True)
    initial_count = len(combined)
    print(f"Total records before cleaning: {initial_count:,}")

    combined = combined[combined["text"].str.strip() != ""].copy()
    empty_removed = initial_count - len(combined)
    if empty_removed > 0:
        print(f"Empty text records removed : {empty_removed:,}")

    print("\n--- Duplicate Articles Check (based on 'text') ---")
    duplicates_before = int(combined.duplicated(subset=["text"]).sum())
    combined = combined.drop_duplicates(subset=["text"]).copy()
    duplicates_removed = duplicates_before
    records_remaining = len(combined)

    print(f"Duplicates before removal : {duplicates_before:,}")
    print(f"Duplicates removed        : {duplicates_removed:,}")
    print(f"Records remaining         : {records_remaining:,}")

    print(f"\n[INFO] Shuffling dataset (random_state={random_state})...")
    shuffled = combined.sample(frac=1.0, random_state=random_state)
    shuffled.reset_index(drop=True, inplace=True)

    return shuffled


def save_dataset(df: pd.DataFrame, output_path: Path | None = None) -> Path:
    """Save DataFrame to CSV file.

    Args:
        df: DataFrame to save.
        output_path: Destination path.

    Returns:
        Path: Path where the dataset was saved.
    """
    if output_path is None:
        project_root = Path(__file__).resolve().parent.parent
        output_path = project_root / "data" / "processed" / "cleaned_news.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[INFO] Saving dataset to: {output_path}")
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[SUCCESS] Dataset successfully saved ({len(df):,} rows).")

    return output_path


def preprocess_text(text: any) -> str:
    """Perform balanced NLP preprocessing on a single news text.

    Pipeline steps:
    1. Safe handling for None, NaN, or non-string inputs.
    2. Lowercasing.
    3. Removal of HTML tags.
    4. Removal of URLs (http, https, www).
    5. Removal of email addresses.
    6. Removal of unnecessary special characters (keeps alphanumeric & space).
    7. Normalization of multiple consecutive spaces into a single space.
    8. Removal of leading and trailing whitespace.

    Args:
        text: Raw input text.

    Returns:
        str: Cleaned text string.
    """
    if text is None or pd.isna(text):
        return ""
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = SPECIAL_CHAR_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()

    return text


def preprocess_dataset(
    input_path: Path | None = None, output_path: Path | None = None
) -> pd.DataFrame:
    """Load cleaned dataset, apply NLP preprocessing, deduplicate, validate, and save.

    Args:
        input_path: Path to input CSV (defaults to data/processed/cleaned_news.csv).
        output_path: Path to output CSV (defaults to data/processed/preprocessed_news.csv).

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    project_root = Path(__file__).resolve().parent.parent

    if input_path is None:
        input_path = project_root / "data" / "processed" / "cleaned_news.csv"

    if output_path is None:
        output_path = project_root / "data" / "processed" / "preprocessed_news.csv"

    df = load_dataset_file(input_path, "Cleaned News Dataset (cleaned_news.csv)")
    orig_total = len(df)

    print(f"\n[INFO] Applying NLP preprocessing pipeline to {orig_total:,} records...")
    df["text"] = df["text"].apply(preprocess_text)

    df = df[df["text"].str.strip() != ""].copy()
    df = df.drop_duplicates(subset=["text"]).copy()
    df.reset_index(drop=True, inplace=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[SUCCESS] Saved {len(df):,} records to {output_path.name}.")

    return df


# ==============================================================================
# STEP 5: STRATIFIED DATASET SPLITTING PIPELINE
# ==============================================================================


def split_dataset(
    df: pd.DataFrame, random_state: int = 42
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Split preprocessed dataset into stratified Train (64%), Val (16%), Test (20%).

    Split hierarchy:
    1. First split: 80% train_temp / 20% test (stratified on label)
    2. Second split: 80% train / 20% val from train_temp (stratified on label)
       -> Yields: 64% Train, 16% Val, 20% Test.

    Strict Data Leakage Controls:
    - Zero overlap in text records across splits.
    - No tokenizer, scaler, or vocabulary fitted here.
    - Preserves stratify proportions in all splits.

    Args:
        df: Preprocessed DataFrame containing 'text' and 'label'.
        random_state: Seed for reproducible splitting (default: 42).

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    if "text" not in df.columns or "label" not in df.columns:
        raise KeyError(
            f"Expected columns ['text', 'label'] in DataFrame, but found: {list(df.columns)}"
        )

    # 2. Separate features and labels
    X = df["text"]
    y = df["label"]
    total_samples = len(df)

    # 3. First split: 80% train_temp, 20% test (stratified)
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=random_state,
        stratify=y,
    )

    # 4. Second split from 80% train_temp: 80% train (64% of total), 20% val (16% of total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_temp,
        y_train_temp,
        test_size=0.20,
        random_state=random_state,
        stratify=y_train_temp,
    )

    # Reset indices for all series
    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # 6. Print dataset split statistics
    print("\n===== DATASET SPLIT =====")
    print(f"\nTotal samples: {total_samples:,}")
    print(f"Training samples: {len(X_train):,} ({len(X_train)/total_samples*100:.2f}%)")
    print(f"Validation samples: {len(X_val):,} ({len(X_val)/total_samples*100:.2f}%)")
    print(f"Testing samples: {len(X_test):,} ({len(X_test)/total_samples*100:.2f}%)")

    # 6. Print class distribution
    print("\n===== CLASS DISTRIBUTION =====")

    splits_data = [
        ("Training", y_train),
        ("Validation", y_val),
        ("Testing", y_test),
    ]

    for name, y_split in splits_data:
        fake_count = int((y_split == 0).sum())
        real_count = int((y_split == 1).sum())
        split_len = len(y_split)
        print(f"\n{name}:")
        print(f"Fake: {fake_count:,} ({fake_count/split_len*100:.2f}%)")
        print(f"Real: {real_count:,} ({real_count/split_len*100:.2f}%)")

    # 7. Verification: All three datasets contain both classes (0 and 1)
    for name, y_split in splits_data:
        unique_classes = set(y_split.unique())
        if unique_classes != {0, 1}:
            raise ValueError(
                f"Validation Error: {name} split must contain both classes {{0, 1}}, found: {unique_classes}"
            )

    # 8. Verification: Length sum equals total dataset size
    total_split_len = len(X_train) + len(X_val) + len(X_test)
    if total_split_len != total_samples:
        raise ValueError(
            f"Validation Error: Sum of splits ({total_split_len}) does not equal total samples ({total_samples})."
        )

    # 9. Verification: Check for duplicate articles across splits (Data Leakage Prevention)
    train_texts = set(X_train)
    val_texts = set(X_val)
    test_texts = set(X_test)

    train_val_overlap = train_texts & val_texts
    train_test_overlap = train_texts & test_texts
    val_test_overlap = val_texts & test_texts

    if train_val_overlap:
        raise ValueError(
            f"Data Leakage Error: Found {len(train_val_overlap)} overlapping texts between Train and Validation!"
        )
    if train_test_overlap:
        raise ValueError(
            f"Data Leakage Error: Found {len(train_test_overlap)} overlapping texts between Train and Test!"
        )
    if val_test_overlap:
        raise ValueError(
            f"Data Leakage Error: Found {len(val_test_overlap)} overlapping texts between Validation and Test!"
        )

    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    """Main execution pipeline for STEP 5."""
    project_root = Path(__file__).resolve().parent.parent
    preprocessed_path = project_root / "data" / "processed" / "preprocessed_news.csv"

    # Ensure preprocessed dataset exists
    if not preprocessed_path.exists():
        print("[INFO] preprocessed_news.csv not found. Running preprocessing first...")
        cleaned_path = project_root / "data" / "processed" / "cleaned_news.csv"
        if not cleaned_path.exists():
            fake_df, true_df = load_datasets()
            fake_df, true_df = create_labels(fake_df, true_df)
            cleaned_df = clean_dataset(fake_df, true_df, random_state=42)
            save_dataset(cleaned_df, cleaned_path)
        preprocess_dataset(input_path=cleaned_path, output_path=preprocessed_path)

    # 1. Load data/processed/preprocessed_news.csv
    df = load_dataset_file(preprocessed_path, "Preprocessed News Dataset")

    # Run stratified splitting and data leakage validations
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df, random_state=42)

    # Final step completion printout
    print("\n" + "=" * 40)
    print("STEP 5 DATASET SPLITTING COMPLETED")
    print("=" * 40)
    print("\nTraining: 64%")
    print("Validation: 16%")
    print("Testing: 20%\n")
    print("Data leakage checks: PASSED\n")


if __name__ == "__main__":
    main()
