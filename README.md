# 🎬 Data Speaks!! — IMDB Sentiment Analysis

Welcome to the **Data Speaks!!** project repository! This project fulfills the course assignment requirements by applying Machine Learning and Deep Learning algorithms to a publicly available educational dataset (IMDB Movie Reviews) to perform binary sentiment classification.

## 📂 Project Structure

To make it easy to evaluate, the project is strictly organized into independent folders based on the pipeline stage and the specific models used.

```text
msd_project/
├── README.md               <-- You are here!
│
├── notebooks/              <-- 1. Data Preprocessing & EDA
│   ├── data_to_csv.ipynb   (Extracts raw IMDB data into usable CSVs)
│   └── EDA.ipynb           (Comprehensive Exploratory Data Analysis)
│
├── processed/              <-- 2. The working dataset
│   ├── imdb_train.csv      (25,000 raw training records)
│   └── imdb_test.csv       (25,000 raw testing records)
│
├── logistic/               <-- 3. Baseline Machine Learning Model
│   └── LOGISTIC.ipynb      (TF-IDF Vectorization + Logistic Regression)
│
├── lstm/                   <-- 4. Deep Learning Sequence Model
│   └── LSTM.ipynb          (Bidirectional LSTM built from scratch with PyTorch)
│
└── bert/                   <-- 5. Advanced Transformer Model
    └── BERT.ipynb          (Fine-tuning bert-base-uncased with PyTorch/Transformers)
```

## 🚀 How to Review the Project

Each modeling notebook (`LOGISTIC.ipynb`, `LSTM.ipynb`, and `BERT.ipynb`) is designed to be **fully self-contained**. 

If you want to review the code execution top-to-bottom, we recommend following this order:

1. **Exploratory Data Analysis**: Open `notebooks/EDA.ipynb`. This notebook proves the dataset is balanced, explores word frequencies/N-grams, and justifies the text-cleaning steps by identifying HTML artifacts.
2. **Baseline Model (TF-IDF)**: Open `logistic/LOGISTIC.ipynb`. This demonstrates a foundational machine learning approach using TF-IDF feature extraction.
3. **Deep Learning Model**: Open `lstm/LSTM.ipynb`. This demonstrates sequence modeling, vocabulary building, and early stopping.
4. **State-of-the-Art Model**: Open `bert/BERT.ipynb`. This demonstrates fine-tuning a pre-trained transformer model using a GPU.

### 🧹 Note on Data Cleaning
Rather than creating a single massive "cleaning" script, the text cleaning (HTML stripping, whitespace normalization, deduplication) is baked directly into **Step 3** of every modeling notebook. This ensures that every model's pipeline can be run completely independently from start to finish.

### ⚠️ Note on Model Weights
Because GitHub has a strict 100 MB per-file limit, the finalized deep learning model weights (e.g., the 400MB+ `.safetensors` for BERT and `.pt` for LSTM) are **not included** in this repository. The CSV datasets and fully executed notebooks *are* included, so all outputs, logs, and plots can be reviewed directly. To regenerate the model weights locally, simply "Run All" on the respective notebooks.

## 📊 Expected Deliverables Covered
- [x] **Dataset Selection**: IMDB Movie Reviews (50,000 records).
- [x] **EDA & Preprocessing**: `notebooks/EDA.ipynb` and Step 3 of model notebooks.
- [x] **Machine Learning Algorithms**: Logistic Regression (Classification).
- [x] **Deep Learning Algorithms**: LSTM and BERT.
- [ ] **Insights & Interpretations**: (To be detailed in the project report & web page).
- [ ] **Responsive Web Page**: (Pending development).
- [ ] **Project Report (3-5 pages)**: (Pending).
