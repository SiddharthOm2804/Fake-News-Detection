"""WELFake Model Prediction Verification Script.

STEP 17: Evaluate existing trained model on actual records from the WELFake dataset.
Tests 10 REAL records (original WELFake 0 -> project 1) and 10 FAKE records (original WELFake 1 -> project 0).
Calculates REAL accuracy, FAKE accuracy, and Overall accuracy.
"""

from pathlib import Path
import os
import pandas as pd
import numpy as np

# Suppress TensorFlow verbose logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from src.predict import predict_news


def evaluate_welfake_predictions(random_state: int = 42):
    """Run verification tests on 10 REAL and 10 FAKE records from WELFake."""
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "processed" / "welfake_cleaned_news.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Error: WELFake dataset not found at '{data_path}'. "
            f"Please run 'python -m src.welfake_preprocessing' first."
        )

    print("=" * 70)
    print("      STEP 17: VERIFY WELFake DATASET PREDICTIONS (NO RETRAINING)")
    print("=" * 70)

    print(f"\n[INFO] Loading prepared WELFake dataset: {data_path.name}")
    df = pd.read_csv(data_path)

    # In welfake_cleaned_news.csv:
    # label 1 = REAL (original WELFake label 0)
    # label 0 = FAKE (original WELFake label 1)
    real_subset = df[df["label"] == 1]
    fake_subset = df[df["label"] == 0]

    print(f"[INFO] Available records -> REAL (1): {len(real_subset):,} | FAKE (0): {len(fake_subset):,}")

    # Select 10 REAL and 10 FAKE records
    real_samples = real_subset.sample(n=10, random_state=random_state).copy()
    fake_samples = fake_subset.sample(n=10, random_state=random_state).copy()

    # Track metrics separately
    real_correct = 0
    real_incorrect = 0

    fake_correct = 0
    fake_incorrect = 0

    # 1. Evaluate REAL Records
    print("\n" + "-" * 70)
    print("PART 1: TESTING 10 REAL RECORDS (Original WELFake: 0 -> Project: 1)")
    print("-" * 70)

    for i, (_, row) in enumerate(real_samples.iterrows(), start=1):
        text = str(row["text"])
        orig_welfake_lbl = 0
        expected_project_lbl = 1
        expected_name = "REAL"

        res = predict_news(text)
        pred_name = res["prediction"]
        real_prob = res["probability"]
        fake_prob = 1.0 - real_prob
        conf = res["confidence"]

        is_correct = (pred_name == expected_name)
        if is_correct:
            real_correct += 1
            tag = "[CORRECT]"
        else:
            real_incorrect += 1
            tag = "[INCORRECT]"

        snippet = text[:110] + "..." if len(text) > 110 else text

        print(f"\nReal Sample #{i} {tag}")
        print(f"Snippet:                \"{snippet}\"")
        print(f"Original WELFake Label: {orig_welfake_lbl} (REAL)")
        print(f"Expected Project Label: {expected_project_lbl} ({expected_name})")
        print(f"Predicted Label:        {pred_name}")
        print(f"REAL Probability:       {real_prob * 100:.2f}%")
        print(f"FAKE Probability:       {fake_prob * 100:.2f}%")
        print(f"Confidence:             {conf * 100:.2f}%")

    # 2. Evaluate FAKE Records
    print("\n" + "-" * 70)
    print("PART 2: TESTING 10 FAKE RECORDS (Original WELFake: 1 -> Project: 0)")
    print("-" * 70)

    for i, (_, row) in enumerate(fake_samples.iterrows(), start=1):
        text = str(row["text"])
        orig_welfake_lbl = 1
        expected_project_lbl = 0
        expected_name = "FAKE"

        res = predict_news(text)
        pred_name = res["prediction"]
        real_prob = res["probability"]
        fake_prob = 1.0 - real_prob
        conf = res["confidence"]

        is_correct = (pred_name == expected_name)
        if is_correct:
            fake_correct += 1
            tag = "[CORRECT]"
        else:
            fake_incorrect += 1
            tag = "[INCORRECT]"

        snippet = text[:110] + "..." if len(text) > 110 else text

        print(f"\nFake Sample #{i} {tag}")
        print(f"Snippet:                \"{snippet}\"")
        print(f"Original WELFake Label: {orig_welfake_lbl} (FAKE)")
        print(f"Expected Project Label: {expected_project_lbl} ({expected_name})")
        print(f"Predicted Label:        {pred_name}")
        print(f"REAL Probability:       {real_prob * 100:.2f}%")
        print(f"FAKE Probability:       {fake_prob * 100:.2f}%")
        print(f"Confidence:             {conf * 100:.2f}%")

    # Metrics Calculation
    real_total = len(real_samples)
    fake_total = len(fake_samples)
    total_samples = real_total + fake_total
    total_correct = real_correct + fake_correct

    real_acc = (real_correct / real_total) * 100.0
    fake_acc = (fake_correct / fake_total) * 100.0
    overall_acc = (total_correct / total_samples) * 100.0

    print("\n" + "=" * 70)
    print("                 WELFake PREDICTION EVALUATION SUMMARY")
    print("=" * 70)

    print("\nREAL SAMPLES (10 articles):")
    print(f"  - Correct predictions   : {real_correct} / {real_total}")
    print(f"  - Incorrect predictions : {real_incorrect} / {real_total}")
    print(f"  - Accuracy              : {real_acc:.2f}%")

    print("\nFAKE SAMPLES (10 articles):")
    print(f"  - Correct predictions   : {fake_correct} / {fake_total}")
    print(f"  - Incorrect predictions : {fake_incorrect} / {fake_total}")
    print(f"  - Accuracy              : {fake_acc:.2f}%")

    print("\nOVERALL METRICS (20 total articles):")
    print(f"  - Total correct         : {total_correct} / {total_samples}")
    print(f"  - Overall Accuracy      : {overall_acc:.2f}%\n")

    print("=" * 70)
    print("ANALYSIS CONCLUSION:")
    if overall_acc >= 75.0:
        print("  The current Kaggle-trained LSTM generalizes reasonably well to WELFake.")
    elif overall_acc >= 50.0:
        print("  The model shows partial domain generalization with noticeable domain shift errors.")
    else:
        print("  The model struggles on WELFake due to significant distribution/vocabulary differences.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    evaluate_welfake_predictions()
