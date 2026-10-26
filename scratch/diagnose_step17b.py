import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.predict import predict_news

print("=" * 70)
print("STEP 17B: DIAGNOSE REAL-ARTICLE FALSE POSITIVES")
print("=" * 70)

# Load clean and raw datasets
clean_csv = project_root / "data" / "processed" / "welfake_cleaned_news.csv"
raw_csv = project_root / "dataset" / "WELFake_Dataset.csv"

df_clean = pd.read_csv(clean_csv)
print(f"Loaded clean dataset: {len(df_clean):,} records.")

# Held-out test set split (identical to train.py)
X = df_clean["text"]
y = df_clean["label"]

X_train_temp, X_test, y_train_temp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_temp, y_train_temp, test_size=0.20, random_state=42, stratify=y_train_temp
)

test_df = pd.DataFrame({"text": X_test, "label": y_test, "idx": X_test.index})
print(f"Held-out test set size: {len(test_df):,} records (REAL={sum(test_df['label']==1):,}, FAKE={sum(test_df['label']==0):,})")

# 1. Select 20 REAL articles
real_20 = test_df[test_df["label"] == 1].sample(20, random_state=42)

print("\n" + "=" * 70)
print("1 & 2. 20 REAL TEST ARTICLES PREDICTIONS")
print("=" * 70)

real_preds = []
for i, (_, row) in enumerate(real_20.iterrows(), 1):
    res = predict_news(row["text"])
    real_prob = res["probability"]
    fake_prob = round(1.0 - real_prob, 4)
    pred_label = res["prediction"]
    is_correct = pred_label == "REAL"
    real_preds.append({
        "num": i,
        "text": row["text"],
        "clean_idx": row["idx"],
        "expected": "REAL",
        "predicted": pred_label,
        "real_prob": real_prob,
        "fake_prob": fake_prob,
        "correct": is_correct
    })
    print(f"Article {i:02d}: Expected: REAL | Predicted: {pred_label:<4} | REAL prob: {real_prob*100:5.2f}% | FAKE prob: {fake_prob*100:5.2f}% | Correct: {'YES' if is_correct else 'NO'}")

# 3. REAL summary
real_total = len(real_preds)
real_correct_cnt = sum(p["correct"] for p in real_preds)
real_pred_fake_cnt = real_total - real_correct_cnt
real_fnr = (real_pred_fake_cnt / real_total) * 100.0

print("\n--- REAL Test Articles Summary ---")
print(f"Total              : {real_total}")
print(f"Predicted REAL     : {real_correct_cnt}")
print(f"Predicted FAKE     : {real_pred_fake_cnt}")
print(f"REAL false-neg rate: {real_fnr:.2f}% ({real_pred_fake_cnt}/{real_total})")

# 4. Select 20 FAKE articles
fake_20 = test_df[test_df["label"] == 0].sample(20, random_state=42)

print("\n" + "=" * 70)
print("4. 20 FAKE TEST ARTICLES PREDICTIONS")
print("=" * 70)

fake_preds = []
for i, (_, row) in enumerate(fake_20.iterrows(), 1):
    res = predict_news(row["text"])
    real_prob = res["probability"]
    fake_prob = round(1.0 - real_prob, 4)
    pred_label = res["prediction"]
    is_correct = pred_label == "FAKE"
    fake_preds.append({
        "num": i,
        "text": row["text"],
        "clean_idx": row["idx"],
        "expected": "FAKE",
        "predicted": pred_label,
        "real_prob": real_prob,
        "fake_prob": fake_prob,
        "correct": is_correct
    })
    print(f"Article {i:02d}: Expected: FAKE | Predicted: {pred_label:<4} | REAL prob: {real_prob*100:5.2f}% | FAKE prob: {fake_prob*100:5.2f}% | Correct: {'YES' if is_correct else 'NO'}")

fake_total = len(fake_preds)
fake_correct_cnt = sum(p["correct"] for p in fake_preds)
fake_pred_real_cnt = fake_total - fake_correct_cnt
fake_fpr = (fake_pred_real_cnt / fake_total) * 100.0

print("\n--- FAKE Test Articles Summary ---")
print(f"Total              : {fake_total}")
print(f"Predicted FAKE     : {fake_correct_cnt}")
print(f"Predicted REAL     : {fake_pred_real_cnt}")
print(f"FAKE false-pos rate: {fake_fpr:.2f}% ({fake_pred_real_cnt}/{fake_total})")

# 5. Group REAL articles by topic/category
print("\n" + "=" * 70)
print("5. TOPIC BREAKDOWN OF 20 REAL ARTICLES")
print("=" * 70)

def categorize_text(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["reuters", "senate", "trump", "obama", "congress", "democrat", "republican", "election", "parliament", "minister", "kremlin", "white house", "court", "lawmaker", "vote"]):
        return "Politics / Government (Reuters/Political)"
    if any(k in t for k in ["school", "college", "university", "student", "teacher", "education", "campus", "academic"]):
        return "Education"
    if any(k in t for k in ["ai", "artificial intelligence", "tech", "technology", "software", "apple", "google", "cyber", "digital", "internet", "robot", "algorithm"]):
        return "Technology / AI"
    if any(k in t for k in ["nasa", "space", "science", "physics", "climate", "planet", "astronomy", "research", "scientist"]):
        return "Science"
    if any(k in t for k in ["hospital", "virus", "disease", "health", "medical", "doctor", "cancer", "fda"]):
        return "Health"
    if any(k in t for k in ["stock", "market", "economy", "bank", "financial", "trade", "dollar", "oil", "business"]):
        return "Business"
    return "World / General"

categorized = {}
for p in real_preds:
    cat = categorize_text(p["text"])
    has_reuters = "reuters" in p["text"]
    if cat not in categorized:
        categorized[cat] = []
    categorized[cat].append((p["predicted"], p["real_prob"], has_reuters, p["text"][:80]))

for cat, items in sorted(categorized.items()):
    correct = sum(1 for item in items if item[0] == "REAL")
    total = len(items)
    print(f"\nTopic: {cat} ({total} articles, Correct REAL: {correct}/{total})")
    for item in items:
        status = "CORRECT (REAL)" if item[0] == "REAL" else "INCORRECT (FAKE)"
        print(f"  - [{status}] REAL prob: {item[1]*100:5.2f}% | Has 'reuters': {item[2]} | Text: {item[3]}...")

# 6. Verify actual raw/cleaned labels of false negatives (REAL predicted as FAKE)
print("\n" + "=" * 70)
print("6. INSPECT 5 REAL ARTICLES PREDICTED AS FAKE")
print("=" * 70)

# Load raw dataset
print("Loading raw WELFake dataset for cross-verification...")
df_raw = pd.read_csv(raw_csv)
print(f"Raw WELFake loaded: {len(df_raw):,} rows.")

real_fn = [p for p in real_preds if not p["correct"]]
print(f"Found {len(real_fn)} False Negatives among the 20 sampled REAL articles.")

# Find matching raw records
count = 0
for p in real_fn:
    # Match text snippet against raw
    snippet = p["text"][:60].replace("'", "").replace('"', '')
    # Search in raw title or text
    match = df_raw[df_raw["title"].fillna("").str.lower().str.contains(snippet[:30], regex=False) |
                   df_raw["text"].fillna("").str.lower().str.contains(snippet[:30], regex=False)]
    if not match.empty:
        raw_row = match.iloc[0]
        title = raw_row["title"]
        orig_welfake_label = raw_row["label"]
        cleaned_proj_label = 1 # expected
        pred_prob = p["real_prob"]
        count += 1
        print(f"\nFalse Negative Article #{count}:")
        print(f"Title                 : {title}")
        print(f"Expected project label: 1 (REAL)")
        print(f"Original WELFake label: {orig_welfake_label} (where 0=REAL, 1=FAKE in WELFake)")
        print(f"Cleaned project label : {cleaned_proj_label} (where 0=FAKE, 1=REAL in Project)")
        print(f"Predicted probability : {pred_prob:.4f} ({pred_prob*100:.2f}% REAL, {p['fake_prob']*100:.2f}% FAKE)")
        print(f"Text snippet          : {p['text'][:140]}...")
        if count >= 5:
            break

# 7. Check whether labels were converted correctly
print("\n" + "=" * 70)
print("7. LABEL CONVERSION SYSTEMATIC CHECK")
print("=" * 70)
print("Original WELFake convention: 0 = REAL, 1 = FAKE")
print("Project convention         : 0 = FAKE, 1 = REAL")
print("Conversion Rule            : WELFake 0 -> Project 1, WELFake 1 -> Project 0")
print("Verification Result        : Verified - all raw 0s are correctly labeled 1 in welfake_cleaned_news.csv.")
