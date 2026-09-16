# Data Speaks!! — Midterm Submission

**Course Assignment:** Data Speaks!!  
**Dataset:** IMDB Large Movie Review Dataset (Binary Sentiment Classification)

This repository serves as the official Midterm Submission, containing all source code, models, data preprocessing pipelines, the final project website, and the project report.

## 📋 Midterm Deliverables Checklist
- [x] **Source code (Python/Jupyter Notebook):** Located in `notebooks/`, `logistic/`, `naive_bayes/`, `lstm/`, `bert/`, and `dim_reduction/`.
- [x] **Dataset with preprocessing steps:** Raw data processing is in `notebooks/data_to_csv.ipynb`. Cleaning steps (HTML stripping, etc.) are implemented across all model notebooks.
- [x] **Final web page:** Located in `docs/` and hosted via GitHub Pages. Features dataset overview, visualizations, methodology, findings, and conclusions.
- [x] **Hosted through GitHub:** Repository is pushed and live.
- [x] **Project report (3–5 pages):** See `Midterm_Project_Report.md`.

## 🗂️ Project Structure (File Names Formatted)

```text
msd_project/
├── Midterm_Project_Report.md  <-- 📄 Official 3-5 Page Project Report
├── README.md                  <-- 📄 You are here!
│
├── docs/                      <-- 🌐 Project Website (Responsive HTML/CSS/JS)
│   ├── index.html             (Dataset Overview & Model Comparisons)
│   ├── eda.html               (Data Visualizations)
│   ├── css/ & js/             (Styling & Interactivity)
│   └── model-*.html           (Methodology & Key Findings for each model)
│
├── notebooks/                 <-- 📊 1. Data Preprocessing & EDA
│   ├── data_to_csv.ipynb      (Data Extraction)
│   └── EDA.ipynb              (Exploratory Data Analysis)
│
├── dim_reduction/             <-- 📉 2. Dimensionality Reduction Task
│   └── DIM_REDUCTION.ipynb    (Truncated SVD / PCA on TF-IDF data)
│
├── naive_bayes/               <-- 🤖 3. Classification Task (Model 1)
│   └── Naive_Bayes_Report.docx (Baseline Naive Bayes implementation)
│
├── logistic/                  <-- 🤖 4. Classification Task (Model 2)
│   └── LOGISTIC.ipynb         (TF-IDF Vectorization + Logistic Regression)
│
├── lstm/                      <-- 🧠 5. Classification Task (Model 3)
│   └── LSTM.ipynb             (Bidirectional LSTM built from scratch)
│
└── bert/                      <-- 🚀 6. Classification Task (Model 4)
    └── BERT.ipynb             (Fine-tuning bert-base-uncased Transformer)
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
- [x] **Responsive Web Page**: Completed (`docs/` folder).
- [ ] **Project Report (3-5 pages)**: (Pending).
