"""Production-ready inference pipeline for Fake News Detection.

STEP 10: Standalone prediction pipeline for new, previously unseen news articles.
Loads:
- models/fake_news_model.keras
- models/tokenizer.pkl
- models/preprocessing_config.json

Provides:
- predict_news(text): Reusable inference function returning class, confidence, and probability.
- Input validation for empty/None/non-string values.
- Zero data leakage, no retraining, and no refitting of tokenizer.
"""

from pathlib import Path
import json
import os
import pickle
import sys
import numpy as np

# Suppress verbose TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.data_preprocessing import preprocess_text

# Cached components for fast consecutive inferences
_CACHED_MODEL = None
_CACHED_TOKENIZER = None
_CACHED_CONFIG = None


def get_project_paths() -> tuple[Path, Path, Path]:
    """Retrieve absolute paths to required model artifacts.

    Returns:
        tuple[Path, Path, Path]: (model_path, tokenizer_path, config_path)
    """
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"
    model_path = models_dir / "fake_news_model.keras"
    tokenizer_path = models_dir / "tokenizer.pkl"
    config_path = models_dir / "preprocessing_config.json"
    return model_path, tokenizer_path, config_path


def load_prediction_components(
    model_path: Path | None = None,
    tokenizer_path: Path | None = None,
    config_path: Path | None = None,
    force_reload: bool = False,
) -> tuple[tf.keras.Model, any, dict]:
    """Load and cache the trained model, tokenizer, and config with error handling.

    Args:
        model_path: Path to fake_news_model.keras.
        tokenizer_path: Path to tokenizer.pkl.
        config_path: Path to preprocessing_config.json.
        force_reload: Reload components even if cached.

    Returns:
        tuple[tf.keras.Model, Tokenizer, dict]: (model, tokenizer, config)

    Raises:
        FileNotFoundError: If any required component is missing with instruction on how to generate it.
    """
    global _CACHED_MODEL, _CACHED_TOKENIZER, _CACHED_CONFIG

    if not force_reload and _CACHED_MODEL is not None and _CACHED_TOKENIZER is not None and _CACHED_CONFIG is not None:
        return _CACHED_MODEL, _CACHED_TOKENIZER, _CACHED_CONFIG

    default_model, default_tokenizer, default_config = get_project_paths()
    model_path = model_path or default_model
    tokenizer_path = tokenizer_path or default_tokenizer
    config_path = config_path or default_config

    # Check for missing components with actionable guidance
    if not model_path.exists():
        raise FileNotFoundError(
            f"\n[ERROR] Model file not found at: '{model_path}'\n"
            f"Please run the training pipeline first:\n"
            f"  python -m src.train\n"
        )

    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"\n[ERROR] Tokenizer file not found at: '{tokenizer_path}'\n"
            f"Please generate the tokenizer using:\n"
            f"  python -m src.train\n"
        )

    if not config_path.exists():
        raise FileNotFoundError(
            f"\n[ERROR] Preprocessing configuration not found at: '{config_path}'\n"
            f"Please generate configuration using:\n"
            f"  python -m src.train\n"
        )

    # 1. Load Preprocessing Config
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 2. Load Tokenizer (pickle)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    # 3. Load Trained Keras Model
    model = tf.keras.models.load_model(model_path)

    _CACHED_MODEL = model
    _CACHED_TOKENIZER = tokenizer
    _CACHED_CONFIG = config

    return model, tokenizer, config


def validate_input(text: any) -> str:
    """Validate user input text and raise user-friendly errors.

    Args:
        text: Raw input to validate.

    Returns:
        str: Validated string.

    Raises:
        ValueError: If input is None, empty, whitespace-only, or invalid type.
    """
    if text is None:
        raise ValueError("Please enter a news article.")

    if not isinstance(text, str):
        text = str(text)

    if not text.strip():
        raise ValueError("Please enter a news article.")

    return text


def predict_news(text: str) -> dict:
    """Classify a raw news article as FAKE or REAL with prediction confidence.

    Pipeline:
    1. Input validation
    2. NLP text cleaning (identical to training preprocessing)
    3. Tokenization into integer sequences via saved Tokenizer
    4. Sequence padding (post-padding to configured max_sequence_length)
    5. Forward inference through saved LSTM model
    6. Decision boundary mapping (0 = FAKE, 1 = REAL) and confidence calculation

    Args:
        text: News headline, body text, or combined article string.

    Returns:
        dict: {
            "prediction": "REAL" | "FAKE",
            "confidence": float,
            "probability": float
        }
    """
    # 7. Input validation
    clean_raw_text = validate_input(text)

    # 1. Load saved components (cached)
    model, tokenizer, config = load_prediction_components()

    max_sequence_length = config.get("max_sequence_length", 300)
    padding = config.get("padding", "post")
    truncating = config.get("truncating", "post")

    # 2. Text preprocessing (exact same pipeline as training)
    processed_text = preprocess_text(clean_raw_text)
    if not processed_text.strip():
        raise ValueError(
            "The entered article contained only characters removed during preprocessing. "
            "Please provide a meaningful news article with alphanumeric words."
        )

    # 3. Tokenization & Sequence Padding
    sequences = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(
        sequences,
        maxlen=max_sequence_length,
        padding=padding,
        truncating=truncating,
    )

    # 4. Model Prediction
    raw_prob = float(model.predict(padded_sequence, verbose=0)[0][0])

    # Mapping: prob >= 0.5 -> REAL (1), prob < 0.5 -> FAKE (0)
    if raw_prob >= 0.5:
        prediction = "REAL"
        confidence = raw_prob
    else:
        prediction = "FAKE"
        confidence = 1.0 - raw_prob

    # 5. Return structured dictionary with floating-point values
    return {
        "prediction": prediction,
        "confidence": round(float(confidence), 4),
        "probability": round(float(raw_prob), 4),
    }


def format_prediction_card(result: dict) -> str:
    """Format prediction output into a clean CLI banner."""
    pred = result["prediction"]
    conf_pct = result["confidence"] * 100.0
    return (
        f"========================================\n"
        f"FAKE NEWS DETECTION\n"
        f"===================\n\n"
        f"Prediction: {pred}\n"
        f"Confidence: {conf_pct:.2f}%\n\n"
        f"========================================"
    )


def main():
    """Main entry point for command-line testing and demonstrations."""
    print("=" * 60)
    print("      STEP 10: FAKE NEWS PREDICTION PIPELINE")
    print("=" * 60)

    # Verify components are accessible
    try:
        model, tokenizer, config = load_prediction_components()
        print("\n[SUCCESS] Model loaded successfully.")
        print("[SUCCESS] Tokenizer loaded successfully.")
        print("[SUCCESS] Preprocessing config loaded successfully.")
    except FileNotFoundError as err:
        print(f"\n{err}")
        sys.exit(1)

    # 9. Test with 2 Demonstration Samples
    print("\n" + "=" * 60)
    print("DEMONSTRATION EXAMPLES (For Pipeline Validation Only)")
    print("=" * 60)

    demo_articles = [
        (
            "DEMONSTRATION 1: Standard News-Style Format",
            "WASHINGTON (Reuters) - The Federal Reserve announced an adjustment to interest rates "
            "following a two-day policy meeting. Central bank officials stated that economic growth "
            "remains steady while inflation metrics continue to approach targeted thresholds."
        ),
        (
            "DEMONSTRATION 2: Sensational / Fabricated News-Style Format",
            "SHOCKING CONSPIRACY EXPOSED: Secret government documents leaked by anonymous insiders "
            "reveal unbelievable cover-up involving major celebrities and underground facilities! "
            "Mainstream media is desperately hiding this truth from the public! SHARE EVERYWHERE!"
        ),
    ]

    for label, text in demo_articles:
        print(f"\n--- {label} ---")
        print(f"Sample Snippet: \"{text[:110]}...\"")
        result = predict_news(text)
        print(format_prediction_card(result))

    # 8. Interactive Command Line Test (if terminal attached)
    if sys.stdin.isatty():
        try:
            print("\nEnter your own news article to test (or press Enter to skip):")
            user_text = input("Enter a news article: ").strip()
            if user_text:
                result = predict_news(user_text)
                print("\n" + format_prediction_card(result))
        except (KeyboardInterrupt, EOFError):
            print("\nInteractive input cancelled.")
        except ValueError as ve:
            print(f"\n[INPUT ERROR] {ve}")

    # Required STEP 10 completion banner
    print("\n" + "=" * 40)
    print("STEP 10 PREDICTION PIPELINE COMPLETED")
    print("=====================================")
    print("\nPrediction function:\nREADY\n")
    print("Model loading:\nSUCCESS\n")
    print("Tokenizer loading:\nSUCCESS\n")
    print("Configuration loading:\nSUCCESS\n")


if __name__ == "__main__":
    main()
