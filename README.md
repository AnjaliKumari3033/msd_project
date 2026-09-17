# Data Speaks!! — Full Project Submission

**Course Assignment:** Data Speaks!!  
**Dataset:** IMDB Large Movie Review Dataset (Binary Classification, Continuous Regression, & Unsupervised Clustering)

This repository serves as the official Full Project Submission, containing all source code, models, data preprocessing pipelines, the project website, and the comprehensive project report.

## 📋 Deliverables Checklist
- [x] **Source code (Python/Jupyter Notebook):** Located in `notebooks/`, `binary_classification_models/`, `regression_models/`, `clustering/`, and `svd/`.
- [x] **Dataset with preprocessing steps:** Three extraction notebooks are provided in `notebooks/` detailing how the binary, regression, and unsupervised data was parsed.
- [x] **Comprehensive EDA:** Side-by-side comparative visualizations for all three datasets (`notebooks/4_COMPREHENSIVE_EDA.ipynb`).
- [x] **Project report (3–5 pages):** See `Midterm_Project_Report.md` (Updated for final submission).
- [x] **Hosted through GitHub:** Repository is pushed and live.

## 🗂️ Project Structure (File Names Formatted)

```text
msd_project/
├── Midterm_Project_Report.md  <-- 📄 Official Project Report (Expanded for Final)
├── README.md                  <-- 📄 You are here!
│
├── docs/                      <-- 🌐 Project Website (Responsive HTML/CSS/JS)
│
├── processed_for_binary_classification/ <-- 💾 Cleaned CSV Data for Classification
├── processed_for_regression/            <-- 💾 Cleaned CSV Data for Regression
├── processed_for_clustering/            <-- 💾 Cleaned CSV Data for Clustering
│
├── notebooks/                 <-- 📊 1. Data Preprocessing & EDA
│   ├── 1_extract_binary_data.ipynb       (Extracts 0/1 labeled data)
│   ├── 2_extract_regression_data.ipynb   (Extracts 1-10 continuous rating data)
│   ├── 3_extract_unsupervised_data.ipynb (Extracts 50k unlabelled reviews)
│   └── 4_COMPREHENSIVE_EDA.ipynb         (Side-by-side EDA, N-Grams, Meta-Features)
│
├── dim_reduction/             <-- 📉 2. Dimensionality Reduction Task
│   └── svd/
│       └── DIM_REDUCTION.ipynb           (Truncated SVD on TF-IDF data)
│
├── binary_classification_models/ <-- 🤖 3. Classification Models (Pos/Neg)
│   ├── naive_bayes/           (Baseline Naive Bayes implementation)
│   ├── logistic/              (TF-IDF Vectorization + Logistic Regression)
│   ├── lstm/                  (Bidirectional LSTM built from scratch)
│   └── bert/                  (Fine-tuning bert-base-uncased Transformer)
│
└── regression_models/           <-- 📈 4. Continuous Regression Models (1-10 Rating)
    └── bert/                  (BERT Fine-Tuned with MSE Loss for Regression)
```

## 🚀 How to Review the Project

Every modeling notebook is designed to be **fully self-contained**. 

1. **Extraction & EDA**: Look through the `notebooks/` folder. The `4_COMPREHENSIVE_EDA.ipynb` file provides deep statistical and visual insights into word distributions, VADER sentiment in the wild, and correlation between punctuation and anger.
2. **Classification**: Review the evolution of our models in `binary_classification_models/`, starting from `naive_bayes` and moving up to `bert`.
3. **Regression**: Review `regression_models/bert/BERT_REGRESSION.ipynb` to see how we altered the Transformer head to predict a continuous 1-10 rating using Mean Squared Error.
4. **Dimensionality Reduction**: Open `svd/DIM_REDUCTION.ipynb` to see how we compressed 5,000 text dimensions into a 2D scatter plot.

### 🧹 Note on Data Cleaning
Rather than creating a single massive "cleaning" script, text cleaning (HTML stripping, whitespace normalization) is baked directly into the data generation phase or the modeling notebooks. This ensures every pipeline can run independently.

### ⚠️ Note on Model Weights
Because GitHub has a strict 100 MB per-file limit, the finalized deep learning model weights (e.g., the 400MB+ `.safetensors` for BERT and `.pt` for LSTM) are **not included** in this repository. The CSV datasets, artifacts, and fully executed notebooks *are* included, so all outputs, metrics, and plots can be reviewed directly.
