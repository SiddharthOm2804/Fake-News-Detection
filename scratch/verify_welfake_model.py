import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.predict import predict_news

project_root = Path(__file__).resolve().parent.parent

# 1. Inspect Dataset
clean_csv = project_root / "data" / "processed" / "welfake_cleaned_news.csv"
raw_csv = project_root / "dataset" / "WELFake_Dataset.csv"

print("=" * 60)
print("1 & 2. VERIFY DATASET & LABEL MAPPING")
print("=" * 60)

df_clean = pd.read_csv(clean_csv)
print(f"Total rows in clean dataset: {len(df_clean):,}")
print(f"Unique labels: {df_clean['label'].unique()}")
print("Label counts in clean dataset:")
print(df_clean["label"].value_counts())

# Random examples
print("\n--- 5 Random REAL Examples (Clean Label = 1) ---")
real_df = df_clean[df_clean["label"] == 1].sample(5, random_state=42)
for i, (_, r) in enumerate(real_df.iterrows(), 1):
    print(f"REAL #{i} (label={r['label']}): {r['text'][:120]}...")

print("\n--- 5 Random FAKE Examples (Clean Label = 0) ---")
fake_df = df_clean[df_clean["label"] == 0].sample(5, random_state=42)
for i, (_, r) in enumerate(fake_df.iterrows(), 1):
    print(f"FAKE #{i} (label={r['label']}): {r['text'][:120]}...")

# Check raw mapping verification
df_raw = pd.read_csv(raw_csv, nrows=5000)
# Look up a known title in raw vs clean
print("\n--- Raw vs Clean Label Mapping Verification ---")
# Sample 2 real from raw (label 0 in raw)
sample_raw_real = df_raw[df_raw["label"] == 0].head(2)
for _, r in sample_raw_real.iterrows():
    title = str(r["title"]).strip().lower()
    match = df_clean[df_clean["text"].str.contains(title[:30], regex=False)]
    if not match.empty:
        print(f"Raw WELFake Label: {r['label']} (REAL) -> Clean Dataset Label: {match.iloc[0]['label']}")

sample_raw_fake = df_raw[df_raw["label"] == 1].head(2)
for _, r in sample_raw_fake.iterrows():
    title = str(r["title"]).strip().lower()
    match = df_clean[df_clean["text"].str.contains(title[:30], regex=False)]
    if not match.empty:
        print(f"Raw WELFake Label: {r['label']} (FAKE) -> Clean Dataset Label: {match.iloc[0]['label']}")

# 3. Verify Splits
print("\n" + "=" * 60)
print("3. VERIFY TRAINING DATA SPLIT")
print("=" * 60)
X = df_clean["text"]
y = df_clean["label"]

X_train_temp, X_test, y_train_temp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_temp, y_train_temp, test_size=0.20, random_state=42, stratify=y_train_temp
)

print(f"Total Dataset Size : {len(df_clean):,}")
print(f"Training Set Size  : {len(X_train):,}")
print(f"  - REAL (1): {(y_train == 1).sum():,} ({(y_train == 1).mean()*100:.2f}%)")
print(f"  - FAKE (0): {(y_train == 0).sum():,} ({(y_train == 0).mean()*100:.2f}%)")
print(f"Validation Set Size: {len(X_val):,}")
print(f"  - REAL (1): {(y_val == 1).sum():,} ({(y_val == 1).mean()*100:.2f}%)")
print(f"  - FAKE (0): {(y_val == 0).sum():,} ({(y_val == 0).mean()*100:.2f}%)")
print(f"Test Set Size      : {len(X_test):,}")
print(f"  - REAL (1): {(y_test == 1).sum():,} ({(y_test == 1).mean()*100:.2f}%)")
print(f"  - FAKE (0): {(y_test == 0).sum():,} ({(y_test == 0).mean()*100:.2f}%)")

# 4 & 5. Verify Tokenizer and Model
print("\n" + "=" * 60)
print("4 & 5. VERIFY TOKENIZER & MODEL SEMANTICS")
print("=" * 60)
with open(project_root / "models" / "tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open(project_root / "models" / "preprocessing_config.json", "r") as f:
    config = json.load(f)

print(f"Tokenizer vocab size config : {config.get('max_vocab_size')}")
print(f"Tokenizer total unique words: {len(tokenizer.word_index):,}")
print(f"Tokenizer OOV token         : {tokenizer.oov_token}")
print(f"Sequence Length config      : {config.get('max_sequence_length')}")

model = tf.keras.models.load_model(project_root / "models" / "fake_news_model.keras")
print("\nModel Summary Layers:")
for l in model.layers:
    print(f"  Layer: {l.name:<25} Output shape: {str(l.output.shape):<20} Activation: {getattr(l, 'activation', 'None')}")

final_layer = model.layers[-1]
final_act = getattr(final_layer, "activation", None)
act_name = final_act.__name__ if final_act else "Unknown"
print(f"\nFinal Layer Name      : {final_layer.name}")
print(f"Final Activation      : {act_name}")
print(f"Model Loss Function   : {model.loss}")

# 6. Run Test Set Evaluation (Sample 2000 from test set for speed or full test set)
print("\n" + "=" * 60)
print("6. RUN TEST-SET EVALUATION")
print("=" * 60)
# Let's evaluate on 2500 samples from held-out test set (stratified) to run quickly
eval_sub_X, _, eval_sub_y, _ = train_test_split(
    X_test, y_test, train_size=2000, random_state=42, stratify=y_test
)
print(f"Evaluating on {len(eval_sub_X):,} test samples...")

sequences = tokenizer.texts_to_sequences(eval_sub_X)
padded = pad_sequences(
    sequences,
    maxlen=config.get("max_sequence_length", 300),
    padding=config.get("padding", "post"),
    truncating=config.get("truncating", "post"),
)

raw_preds = model.predict(padded, batch_size=128, verbose=0).flatten()
binary_preds = (raw_preds >= 0.5).astype(int)

acc = accuracy_score(eval_sub_y, binary_preds)
prec = precision_score(eval_sub_y, binary_preds)
rec = recall_score(eval_sub_y, binary_preds)
f1 = f1_score(eval_sub_y, binary_preds)
cm = confusion_matrix(eval_sub_y, binary_preds)

print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-Score : {f1:.4f}")
print("\nConfusion Matrix (Rows = True, Cols = Pred):")
print("               Pred FAKE (0)    Pred REAL (1)")
print(f"True FAKE (0)       {cm[0][0]:<12}     {cm[0][1]:<12}")
print(f"True REAL (1)       {cm[1][0]:<12}     {cm[1][1]:<12}")

print(f"\nREAL predicted REAL: {cm[1][1]}")
print(f"REAL predicted FAKE: {cm[1][0]}")
print(f"FAKE predicted FAKE: {cm[0][0]}")
print(f"FAKE predicted REAL: {cm[0][1]}")

print("\nClassification Report:")
print(classification_report(eval_sub_y, binary_preds, target_names=["FAKE (0)", "REAL (1)"]))

# 7. Sanity Check on 10 REAL and 10 FAKE from Test Set
print("\n" + "=" * 60)
print("7. SANITY CHECK (10 REAL & 10 FAKE from Test Set)")
print("=" * 60)

test_df = pd.DataFrame({"text": X_test, "label": y_test})
real_10 = test_df[test_df["label"] == 1].sample(10, random_state=42)
fake_10 = test_df[test_df["label"] == 0].sample(10, random_state=42)

real_correct = 0
print("--- 10 REAL TEST SAMPLES ---")
for i, (_, row) in enumerate(real_10.iterrows(), 1):
    res = predict_news(row["text"])
    pred = res["prediction"]
    prob = res["probability"]
    is_correct = pred == "REAL"
    if is_correct:
        real_correct += 1
    print(f"REAL Sample #{i}:")
    print(f"  Expected: REAL | Predicted: {pred} | Probability: {prob:.4f} ({prob*100:.1f}%) | Correct: {'YES' if is_correct else 'NO'}")

fake_correct = 0
print("\n--- 10 FAKE TEST SAMPLES ---")
for i, (_, row) in enumerate(fake_10.iterrows(), 1):
    res = predict_news(row["text"])
    pred = res["prediction"]
    prob = res["probability"]
    is_correct = pred == "FAKE"
    if is_correct:
        fake_correct += 1
    print(f"FAKE Sample #{i}:")
    print(f"  Expected: FAKE | Predicted: {pred} | Probability: {prob:.4f} ({prob*100:.1f}%) | Correct: {'YES' if is_correct else 'NO'}")

print(f"\nREAL correct: {real_correct}/10")
print(f"FAKE correct: {fake_correct}/10")

# Test a NASA article
print("\n--- NASA PRESS RELEASE TEST ---")
nasa_article = (
    "NASA Space Launch System and Orion Spacecraft Complete Rollout to Launch Pad 39B at Kennedy Space Center. "
    "NASA's Space Launch System rocket and Orion spacecraft arrived at Launch Pad 39B at the agency's Kennedy Space Center in Florida "
    "ahead of the Artemis I test flight. Teams worked through the overnight hours to prepare the mega Moon rocket for final testing."
)
nasa_res = predict_news(nasa_article)
print("NASA Article Prediction:")
print(f"  Prediction : {nasa_res['prediction']}")
print(f"  Probability: {nasa_res['probability']}")
print(f"  Confidence : {nasa_res['confidence']}")
