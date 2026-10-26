"""Model training and LSTM architecture pipeline for Fake News Detection.

STEP 8: Complete Model Training Pipeline.
Handles:
- Dataset loading & stratified 64/16/20 split (X_test held out strictly)
- Tokenization fitted exclusively on X_train
- Sequence padding & configuration serialization
- Model creation and compilation (Adam, binary_crossentropy, accuracy)
- Callbacks: EarlyStopping, ModelCheckpoint (.keras), ReduceLROnPlateau
- Model training on X_train / X_val
- Training history logging (JSON)
- Matplotlib training loss and accuracy plots (PNG)
"""

from pathlib import Path
import json
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Suppress verbose TensorFlow logging for clean console output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import Sequential, layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.data_preprocessing import load_dataset_file, split_dataset


# ==============================================================================
# CONFIGURATION & HYPERPARAMETERS
# ==============================================================================
# Tokenization & Sequence parameters
MAX_VOCAB_SIZE = 20000
MAX_SEQUENCE_LENGTH = 300
PADDING_TYPE = "post"
TRUNCATING_TYPE = "post"
OOV_TOKEN = "<OOV>"

# Model Architecture parameters
EMBEDDING_DIM = 128
SPATIAL_DROPOUT = 0.2
LSTM_UNITS = 128
DROPOUT_RATE = 0.3
DENSE_UNITS = 64
LEARNING_RATE = 0.001

# Training parameters
EPOCHS = 10
BATCH_SIZE = 64


# ==============================================================================
# TOKENIZATION & ENCODING FUNCTIONS
# ==============================================================================


def create_tokenizer(
    X_train: pd.Series,
    max_vocab_size: int = MAX_VOCAB_SIZE,
    oov_token: str = OOV_TOKEN,
) -> Tokenizer:
    """Create and fit Keras Tokenizer strictly on training texts (X_train only)."""
    tokenizer = Tokenizer(
        num_words=max_vocab_size,
        oov_token=oov_token,
    )
    tokenizer.fit_on_texts(X_train)
    return tokenizer


def encode_texts(
    tokenizer: Tokenizer,
    texts: pd.Series | list[str],
    max_length: int = MAX_SEQUENCE_LENGTH,
    padding: str = PADDING_TYPE,
    truncating: str = TRUNCATING_TYPE,
) -> np.ndarray:
    """Transform text sequences into padded integer arrays."""
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(
        sequences,
        maxlen=max_length,
        padding=padding,
        truncating=truncating,
    )


def save_tokenizer(tokenizer: Tokenizer, output_path: Path) -> Path:
    """Serialize and save Tokenizer to disk using pickle."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[SUCCESS] Tokenizer saved to: {output_path}")
    return output_path


def save_config(config_dict: dict, output_path: Path) -> Path:
    """Save preprocessing configuration to JSON for reproducible inference."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)
    print(f"[SUCCESS] Configuration saved to: {output_path}")
    return output_path


# ==============================================================================
# MODEL ARCHITECTURE & COMPILATION
# ==============================================================================


def build_model(
    vocab_size: int = MAX_VOCAB_SIZE,
    max_sequence_length: int = MAX_SEQUENCE_LENGTH,
    embedding_dim: int = EMBEDDING_DIM,
    spatial_dropout: float = SPATIAL_DROPOUT,
    lstm_units: int = LSTM_UNITS,
    dropout_rate: float = DROPOUT_RATE,
    dense_units: int = DENSE_UNITS,
) -> Sequential:
    """Build the LSTM Deep Learning model for Fake News classification.

    Architecture:
    Input -> Embedding -> SpatialDropout1D -> LSTM -> Dropout -> Dense -> Sigmoid
    """
    model = Sequential(
        [
            # Input: Specifies shape and integer datatype of padded token sequences
            layers.Input(
                shape=(max_sequence_length,),
                dtype="int32",
                name="input_sequence",
            ),
            # Embedding: Converts token IDs into dense vector representations.
            layers.Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                name="embedding_layer",
            ),
            # SpatialDropout1D: Helps reduce overfitting in text representations.
            layers.SpatialDropout1D(
                rate=spatial_dropout,
                name="spatial_dropout_1d",
            ),
            # LSTM: Learns sequential dependencies in the news text.
            layers.LSTM(
                units=lstm_units,
                return_sequences=False,
                name="lstm_layer",
            ),
            # Dropout: Regularizes the network.
            layers.Dropout(
                rate=dropout_rate,
                name="dropout_layer",
            ),
            # Dense: Learns higher-level classification features.
            layers.Dense(
                units=dense_units,
                activation="relu",
                name="dense_feature_layer",
            ),
            # Sigmoid: Produces a probability between 0 and 1 (0 = FAKE, 1 = REAL).
            layers.Dense(
                units=1,
                activation="sigmoid",
                name="output_sigmoid",
            ),
        ],
        name="Fake_News_LSTM_Detector",
    )
    return model


def compile_model(
    model: Sequential,
    learning_rate: float = LEARNING_RATE,
) -> Sequential:
    """Compile the model with Adam optimizer, binary crossentropy, and accuracy."""
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ==============================================================================
# STEP 8: TRAINING PIPELINE, CALLBACKS & VISUALIZATION
# ==============================================================================


def train_model(
    model: Sequential,
    X_train: np.ndarray,
    y_train: np.ndarray | pd.Series,
    X_val: np.ndarray,
    y_val: np.ndarray | pd.Series,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    model_save_path: Path | None = None,
) -> tf.keras.callbacks.History:
    """Execute training pipeline with EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau.

    CRITICAL DATA LEAKAGE PREVENTION:
    X_test and y_test are NEVER passed into this function or seen during training.

    Args:
        model: Compiled Keras Sequential model.
        X_train: Padded training token sequences.
        y_train: Training labels (0 = FAKE, 1 = REAL).
        X_val: Padded validation token sequences.
        y_val: Validation labels.
        epochs: Max number of training epochs.
        batch_size: Mini-batch size.
        model_save_path: Path to save the best performing model (.keras).

    Returns:
        History: Training history object.
    """
    if model_save_path is None:
        project_root = Path(__file__).resolve().parent.parent
        model_save_path = project_root / "models" / "fake_news_model.keras"

    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. EarlyStopping: halt training when validation loss stops improving
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
        verbose=1,
    )

    # 2. ModelCheckpoint: save strictly the best model based on validation loss
    checkpoint = ModelCheckpoint(
        filepath=str(model_save_path),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    # 3. ReduceLROnPlateau: decay learning rate when improvement plateaus
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=1,
        min_lr=1e-6,
        verbose=1,
    )

    callbacks_list = [early_stopping, checkpoint, reduce_lr]

    # Convert y series to numpy arrays if needed
    y_train_arr = np.asarray(y_train, dtype=np.float32)
    y_val_arr = np.asarray(y_val, dtype=np.float32)

    history = model.fit(
        X_train,
        y_train_arr,
        validation_data=(X_val, y_val_arr),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        verbose=1,
    )

    return history


def save_training_history(history_dict: dict, output_path: Path) -> Path:
    """Save training history metrics to JSON.

    Args:
        history_dict: History dictionary from Keras training.
        output_path: Path to save JSON file.

    Returns:
        Path: Saved file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable_history = {
        key: [float(val) for val in values] for key, values in history_dict.items()
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_history, f, indent=2)
    print(f"[SUCCESS] Training history saved to: {output_path}")
    return output_path


def plot_training_metrics(history_dict: dict, output_dir: Path) -> tuple[Path, Path]:
    """Generate and save separate Matplotlib plots for Accuracy and Loss.

    Args:
        history_dict: History dictionary containing loss, accuracy, val_loss, val_accuracy.
        output_dir: Directory where the PNG plots will be saved.

    Returns:
        tuple[Path, Path]: Paths to (accuracy_plot, loss_plot).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs_range = range(1, len(history_dict["loss"]) + 1)

    # Plot 1: Training Accuracy vs Validation Accuracy
    acc_path = output_dir / "training_accuracy.png"
    plt.figure(figsize=(8, 6))
    plt.plot(
        epochs_range,
        history_dict["accuracy"],
        label="Training Accuracy",
        marker="o",
        color="#1f77b4",
        linewidth=2,
    )
    plt.plot(
        epochs_range,
        history_dict["val_accuracy"],
        label="Validation Accuracy",
        marker="s",
        color="#ff7f0e",
        linewidth=2,
    )
    plt.title("Training Accuracy vs Validation Accuracy", fontsize=14, pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(acc_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Training accuracy plot saved to: {acc_path}")

    # Plot 2: Training Loss vs Validation Loss
    loss_path = output_dir / "training_loss.png"
    plt.figure(figsize=(8, 6))
    plt.plot(
        epochs_range,
        history_dict["loss"],
        label="Training Loss",
        marker="o",
        color="#2ca02c",
        linewidth=2,
    )
    plt.plot(
        epochs_range,
        history_dict["val_loss"],
        label="Validation Loss",
        marker="s",
        color="#d62728",
        linewidth=2,
    )
    plt.title("Training Loss vs Validation Loss", fontsize=14, pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(loc="upper right", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(loss_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Training loss plot saved to: {loss_path}")

    return acc_path, loss_path


def main():
    """Main execution entry point for Model Training on WELFake."""
    project_root = Path(__file__).resolve().parent.parent

    data_path = project_root / "data" / "processed" / "welfake_cleaned_news.csv"
    models_dir = project_root / "models"
    processed_dir = project_root / "data" / "processed"

    models_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_path = models_dir / "tokenizer.pkl"
    config_path = models_dir / "preprocessing_config.json"
    best_model_path = models_dir / "fake_news_model.keras"
    history_json_path = processed_dir / "training_history.json"

    # 1. Load preprocessed WELFake dataset
    df = load_dataset_file(data_path, "Cleaned WELFake Dataset")

    # Step 10 Verification: Verify labels before training
    unique_labels = set(df["label"].unique())
    if unique_labels != {0, 1}:
        raise ValueError(
            f"Fatal Label Mapping Error: Expected {{0, 1}}, got {unique_labels}. "
            "Stopping training immediately."
        )
    print("\n[VERIFICATION] Target Labels Verified:")
    print("  - label 0 = FAKE")
    print("  - label 1 = REAL")

    # 2. Split dataset: 64% Train, 16% Val, 20% Test (stratified, random_state=42)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df, random_state=42)

    # 3. Create tokenizer fitted strictly on X_train ONLY
    tokenizer = create_tokenizer(
        X_train,
        max_vocab_size=MAX_VOCAB_SIZE,
        oov_token=OOV_TOKEN,
    )

    # Save tokenizer and config
    save_tokenizer(tokenizer, tokenizer_path)
    config_data = {
        "max_vocab_size": MAX_VOCAB_SIZE,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "padding": PADDING_TYPE,
        "truncating": TRUNCATING_TYPE,
        "oov_token": OOV_TOKEN,
    }
    save_config(config_data, config_path)

    # 4. Encode sequences for Train, Val, and Test
    X_train_pad = encode_texts(tokenizer, X_train, max_length=MAX_SEQUENCE_LENGTH)
    X_val_pad = encode_texts(tokenizer, X_val, max_length=MAX_SEQUENCE_LENGTH)
    # Note: X_test_pad is prepared for held-out evaluation, strictly excluded from training
    X_test_pad = encode_texts(tokenizer, X_test, max_length=MAX_SEQUENCE_LENGTH)

    # 5. Build and compile LSTM model
    model = build_model(
        vocab_size=MAX_VOCAB_SIZE,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        embedding_dim=EMBEDDING_DIM,
        spatial_dropout=SPATIAL_DROPOUT,
        lstm_units=LSTM_UNITS,
        dropout_rate=DROPOUT_RATE,
        dense_units=DENSE_UNITS,
    )
    model = compile_model(model, learning_rate=LEARNING_RATE)

    # 6. Training Information Display
    print("\n" + "=" * 50)
    print("STARTING MODEL TRAINING ON WELFake DATASET")
    print("==========================================")
    print(f"Training samples   : {len(X_train):,}")
    print(f"Validation samples : {len(X_val):,}")
    print(f"Testing samples    : {len(X_test):,}\n")
    print("Class Distribution:")
    print(f"  - Total Dataset  : FAKE (0) = {int((df['label'] == 0).sum()):,} ({float((df['label'] == 0).mean()*100):.2f}%), REAL (1) = {int((df['label'] == 1).sum()):,} ({float((df['label'] == 1).mean()*100):.2f}%)")
    print(f"  - Training Set   : FAKE (0) = {int((y_train == 0).sum()):,} ({float((y_train == 0).mean()*100):.2f}%), REAL (1) = {int((y_train == 1).sum()):,} ({float((y_train == 1).mean()*100):.2f}%)")
    print(f"  - Validation Set : FAKE (0) = {int((y_val == 0).sum()):,} ({float((y_val == 0).mean()*100):.2f}%), REAL (1) = {int((y_val == 1).sum()):,} ({float((y_val == 1).mean()*100):.2f}%)")
    print(f"  - Testing Set    : FAKE (0) = {int((y_test == 0).sum()):,} ({float((y_test == 0).mean()*100):.2f}%), REAL (1) = {int((y_test == 1).sum()):,} ({float((y_test == 1).mean()*100):.2f}%)\n")
    print(f"Vocabulary size         : {MAX_VOCAB_SIZE:,}")
    print(f"Maximum sequence length : {MAX_SEQUENCE_LENGTH}")
    print(f"Batch size              : {BATCH_SIZE}")
    print(f"Maximum epochs          : {EPOCHS}")
    print("=" * 50 + "\n")

    # 7. Train model (Strictly X_train and X_val, held-out test set NEVER passed)
    history = train_model(
        model=model,
        X_train=X_train_pad,
        y_train=y_train,
        X_val=X_val_pad,
        y_val=y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        model_save_path=best_model_path,
    )

    # 8. Save history to JSON
    save_training_history(history.history, history_json_path)

    # 9. Plot training loss and accuracy
    plot_training_metrics(history.history, processed_dir)

    # 10. Compute best metrics
    epochs_completed = len(history.history["loss"])
    best_val_loss = min(history.history["val_loss"])
    best_val_acc = max(history.history["val_accuracy"])

    # Relative paths for display
    rel_model_path = "models/fake_news_model.keras"
    rel_history_path = "data/processed/training_history.json"

    # 11. Final training summary
    print("\n" + "=" * 50)
    print("TRAINING COMPLETED ON WELFake DATASET")
    print("=====================================")
    print(f"Training samples   : {len(X_train):,}")
    print(f"Validation samples : {len(X_val):,}")
    print(f"Test samples       : {len(X_test):,}")
    print("\nClass Distribution:")
    print(f"  - Total FAKE (0) : {int((df['label'] == 0).sum()):,} ({float((df['label'] == 0).mean()*100):.2f}%)")
    print(f"  - Total REAL (1) : {int((df['label'] == 1).sum()):,} ({float((df['label'] == 1).mean()*100):.2f}%)")
    print(f"  - Train FAKE (0) : {int((y_train == 0).sum()):,}, Train REAL (1): {int((y_train == 1).sum()):,}")
    print(f"  - Val FAKE (0)   : {int((y_val == 0).sum()):,}, Val REAL (1): {int((y_val == 1).sum()):,}")
    print(f"  - Test FAKE (0)  : {int((y_test == 0).sum()):,}, Test REAL (1): {int((y_test == 1).sum()):,}")
    print(f"\nEpochs actually completed : {epochs_completed}")
    print(f"Best validation accuracy  : {best_val_acc:.4f} ({best_val_acc * 100:.2f}%)")
    print(f"Best validation loss      : {best_val_loss:.4f}\n")
    print(f"Best model saved to       : {rel_model_path}")
    print(f"Tokenizer saved to        : models/tokenizer.pkl")
    print(f"Config saved to           : models/preprocessing_config.json")
    print(f"Training history saved to : {rel_history_path}")
    print("=" * 50)

    print("\n" + "=" * 50)
    print("WELFake dataset successfully integrated.")
    print("Model successfully retrained.")
    print("Tokenizer successfully regenerated.")
    print("Prediction artifacts successfully regenerated.")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
