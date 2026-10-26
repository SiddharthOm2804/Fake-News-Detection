# Fake News Detection Using Deep Learning

An end-to-end Deep Learning system built with Python, TensorFlow/Keras, and Streamlit to classify news articles as **FAKE** or **REAL** using Long Short-Term Memory (LSTM) recurrent neural networks and Natural Language Processing (NLP).

---

## 📌 Project Overview

This project classifies news articles based on linguistic, stylistic, and sequential textual patterns learned from historical data. 

> **Important Note:** This system is a statistical machine-learning classification tool. It **does NOT** independently fact-check or verify the real-world factual accuracy of claims. Predictions reflect how closely a text resembles patterns found in the training corpus.

---

## ✨ Key Features

- **LSTM Deep Learning Architecture**: Captures recurrent word dependencies and long-term contextual patterns.
- **NLP Text Preprocessing Pipeline**: Strips HTML tags, URLs, email addresses, and noisy characters while preserving critical semantic words, numbers, and negations.
- **Data-Leakage Prevention**: Strictly stratified splits (64% Train, 16% Validation, 20% Held-Out Test); tokenizer fitted strictly on training data only.
- **Configurable Tokenization & Padding**: Preprocessing hyperparameter logging ensures identical sequence representation at inference time.
- **Comprehensive Evaluation**: Reports test Accuracy, Precision, Recall, F1-score, and generates a pure Matplotlib Confusion Matrix heatmap.
- **Confidence & Probability Breakdown**: Outputs continuous model probabilities and confidence scores without artificial inflation.
- **Production CLI Tool**: Interactive terminal interface (`python -m src.predict`) for rapid command-line testing.
- **Modern Light-Mode Streamlit Dashboard**: Polished SaaS-style interface designed specifically with clean white cards, indigo accents, and responsive layout.

---

## 🛠️ Technology Stack

- **Language**: Python (3.10+)
- **Deep Learning Framework**: TensorFlow / Keras (LSTM, Embedding, SpatialDropout1D)
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning & Evaluation**: Scikit-learn
- **Visualization**: Matplotlib
- **Web Application**: Streamlit

---

## 🧠 Model Architecture

The deep learning pipeline processes raw news text through the following sequential stages:

```
Raw News Article Text
          ↓
[NLP Preprocessing]          (Lowercase, strip HTML/URLs/emails, normalize whitespace)
          ↓
[Tokenization & Padding]     (Keras Tokenizer, max vocab 20,000, post-padded to 300 tokens)
          ↓
[Embedding Layer]            (Dense vector representations: 128 dimensions)
          ↓
[SpatialDropout1D]           (Rate: 0.2, regularizes token feature maps)
          ↓
[LSTM Layer]                 (128 recurrent units, captures contextual sequence order)
          ↓
[Dropout Layer]              (Rate: 0.3, prevents overfitting)
          ↓
[Dense Layer]                (64 units with ReLU activation)
          ↓
[Dense Sigmoid Output]       (1 unit, outputs probability p ∈ [0, 1])
          ↓
Decision Mapping:  p < 0.5 → FAKE (0)  |  p ≥ 0.5 → REAL (1)
```

---

## 📊 Dataset

This project utilizes the popular Kaggle **Fake and Real News Dataset**:
- **Dataset Link**: [Kaggle: Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- **Files**:
  - `dataset/Fake.csv` (~23,481 records): Fabricated, misleading, or satirical news articles.
  - `dataset/True.csv` (~21,417 records): Verified factual news articles (e.g., Reuters wire reports).
- **Target Labels**:
  - `0 = FAKE`
  - `1 = REAL`

---

## 📁 Project Structure

```
FakeNews/
│
├── dataset/
│   ├── Fake.csv                       # Original Kaggle Fake news dataset
│   └── True.csv                       # Original Kaggle Real news dataset
│
├── data/
│   ├── raw/                           # Raw data storage
│   └── processed/                     # Cleaned datasets and training outputs
│       ├── cleaned_news.csv           # Deduplicated and labeled dataset
│       ├── preprocessed_news.csv      # NLP-cleaned dataset
│       ├── training_history.json      # Epoch loss and accuracy metrics
│       ├── training_accuracy.png      # Matplotlib accuracy graph
│       ├── training_loss.png          # Matplotlib loss graph
│       ├── evaluation_results.json    # Test evaluation metrics
│       └── confusion_matrix.png       # Test confusion matrix heatmap
│
├── models/
│   ├── fake_news_model.keras          # Best trained LSTM weights
│   ├── tokenizer.pkl                  # Serialized Keras Tokenizer
│   └── preprocessing_config.json      # Preprocessing inference configuration
│
├── notebooks/                         # Exploratory Jupyter notebooks
│
├── src/
│   ├── __init__.py                    # Source package initializer
│   ├── data_preprocessing.py          # Data inspection, cleaning, and splitting
│   ├── train.py                       # LSTM construction and training pipeline
│   ├── evaluate.py                    # Evaluation metrics and confusion matrix
│   └── predict.py                     # Standalone prediction engine
│
├── app.py                             # Streamlit modern Light-Mode dashboard
├── requirements.txt                   # Dependency specifications
├── README.md                          # Project documentation
└── .gitignore                         # Git exclusion rules
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Fake-News-Detection.git
cd Fake-News-Detection
```

### 2. Create and Activate Virtual Environment
On Windows:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

On macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Dataset Setup
Ensure `Fake.csv` and `True.csv` are placed in the `dataset/` directory:
```
dataset/
├── Fake.csv
└── True.csv
```

---

## 🚀 Running the Project

### Step 1: Preprocess Dataset & Inspect
```bash
python -m src.data_preprocessing
```

### Step 2: Train the LSTM Model
```bash
python -m src.train
```
*Saves the best model to `models/fake_news_model.keras` and records training plots in `data/processed/`.*

### Step 3: Evaluate on Held-Out Test Set
```bash
python -m src.evaluate
```
*Computes test accuracy, precision, recall, F1, and saves `confusion_matrix.png`.*

### Step 4: Run CLI Prediction
```bash
python -m src.predict
```

### Step 5: Launch Streamlit Web Application
```bash
streamlit run app.py
```
*(Or on Windows if `streamlit` is not in your global PATH: `python -m streamlit run app.py`)*
*Access the local web dashboard at `http://localhost:8501`.*

---

## 📈 Evaluation & Performance

The model was evaluated strictly on the **20% unseen, held-out test split** (7,766 articles) with zero data leakage:

| Metric | Score | Percentage |
| :--- | :--- | :--- |
| **Accuracy** | `0.9898` | **98.98%** |
| **Precision** | `0.9933` | **99.33%** |
| **Recall** | `0.9878` | **98.78%** |
| **F1-Score** | `0.9905` | **99.05%** |
| **Total Test Samples** | `7,766` | 100.0% |
| **Correct Predictions** | `7,687` | **98.98%** |
| **Incorrect Predictions** | `79` | **1.02%** |

### Confusion Matrix Breakdown
- **True Fake Identified (True Negatives)**: 3,552 / 3,580 (99.22%)
- **True Real Identified (True Positives)**: 4,135 / 4,186 (98.78%)
- **False Positives (Fake predicted as Real)**: 28
- **False Negatives (Real predicted as Fake)**: 51

*(Empirical metrics saved automatically to [data/processed/evaluation_results.json](file:///c:/Users/SIDDHARTH/Documents/FakeNews/data/processed/evaluation_results.json) and visualized in [data/processed/confusion_matrix.png](file:///c:/Users/SIDDHARTH/Documents/FakeNews/data/processed/confusion_matrix.png).)*

---

## 🖼️ Application Screenshots

*(Features: Interactive Three.js 3D Neural Lattice Visualizer + Cyber-Dark Glassmorphic UI)*

```
+-------------------------------------------------------------------------------+
|  FAKE NEWS DETECTION                                                         |
|  AI-powered news credibility classification using Deep Learning              |
+-------------------------------------------------------------------------------+
|  [ Analyze a News Article ]                                                   |
|  +-------------------------------------------------------------------------+  |
|  | Paste a news article here...                                            |  |
|  +-------------------------------------------------------------------------+  |
|  [ Load Example 1 ]  [ Load Example 2 ]  [ Load Example 3 ]                   |
|  [ Analyze News ]                                                             |
|                                                                               |
|  +-- Likely Real News -----------------------------------------------------+  |
|  | Confidence: 92.69%                                                      |  |
|  | Real Probability: [==============================] 92.69%               |  |
|  | Fake Probability: [==] 7.31%                                            |  |
|  +-------------------------------------------------------------------------+  |
+-------------------------------------------------------------------------------+
```

---

## ⚠️ Limitations

- **Dataset Specificity**: The model reflects linguistic patterns of the Kaggle training dataset (political reporting up to 2018).
- **Not Real-Time Fact-Checking**: The classifier evaluates style and word associations, not objective factual truth.
- **Domain Shifts**: Articles from non-English sources, regional journalism, or novel topics outside the training vocabulary may produce lower accuracy.
- **Confidence Interpretation**: Reported confidence is the statistical probability output by the sigmoid function, not an epistemological certainty.

---

## 🔮 Future Improvements

- **Transformer Backbones**: Integrate BERT / RoBERTa / DeBERTa fine-tuning for deeper semantic understanding.
- **Multilingual Support**: Extend classification to multilingual news corpora via XLM-RoBERTa.
- **Explainable AI (XAI)**: Implement SHAP or LIME token-attribution heatmaps to highlight words driving the classification.
- **Source & Metadata Fusion**: Incorporate publisher domain verification and author credibility scoring.
- **Automated Claim Verification**: Connect with live fact-checking APIs (e.g., Google Fact Check Tools API).

---

## ⚖️ Disclaimer

> This project is intended for educational and research purposes. The prediction represents the model's learned classification patterns and should not be treated as proof that a news article is factually true or false.

---

## 👤 Author

**Developed by Siddharth Om**
#   F a k e - N e w s - D e t e c t i o n  
 #   F a k e - N e w s - D e t e c t i o n  
 #   F a k e - N e w s - D e t e c t i o n  
 