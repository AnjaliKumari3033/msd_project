# Data Speaks!! — Midterm Project Report

**Course:** Data Speaks!!  
**Task:** Binary Sentiment Classification on the IMDB Large Movie Review Dataset  
**Algorithms Applied:** Naive Bayes, Logistic Regression, LSTM (Recurrent Neural Network), BERT (Transformer), and Truncated SVD (Dimensionality Reduction).  

---

## 1. Introduction and Dataset Overview

### 1.1 Project Objective
The objective of this project is to build a robust pipeline capable of classifying the underlying sentiment of natural language text. By analyzing the raw textual content of movie reviews, our models must predict whether the reviewer's sentiment is **Positive (1)** or **Negative (0)**. 

### 1.2 Dataset Selection
We utilized the benchmark **Large Movie Review Dataset (IMDB)**. The dataset contains 50,000 highly polar movie reviews, perfectly balanced between 25,000 positive and 25,000 negative samples. 
* **Training Set:** 25,000 labeled reviews (12.5k Pos / 12.5k Neg)
* **Testing Set:** 25,000 labeled reviews (12.5k Pos / 12.5k Neg)
* **Unsupervised Set:** 50,000 unlabeled reviews (Intentionally ignored during classification training to maintain strict supervised boundaries, but useful for theoretical advanced vocabulary building).

This perfectly balanced nature of the dataset eliminates the need for advanced class-balancing techniques (like SMOTE) and allows us to rely on simple Accuracy as our primary evaluation metric.

---

## 2. Data Preprocessing and Exploratory Data Analysis (EDA)

Before feeding the text to any machine learning algorithm, an extensive data audit and cleaning pipeline was required. Natural language scraped from the web is inherently noisy and unstructured.

### 2.1 Text Cleaning Pipeline
Upon analyzing the raw text, it was discovered that over 50% of the reviews contained HTML artifacts (specifically `<br />` tags) injected during the web-scraping process. A systematic preprocessing pipeline was applied across all models:
1. **HTML Unescaping:** Translating HTML entities (like `&amp;`) back to standard text.
2. **Tag Stripping:** Using Regular Expressions (`re.sub`) to permanently remove line break tags and rogue HTML elements.
3. **Whitespace Normalization:** Ensuring consistent single-spaces between tokens.
4. **Lowercasing & Stemming:** (Used specifically in the baseline Naive Bayes model) to collapse vocabulary variants (e.g., *running* -> *run*).

### 2.2 Exploratory Data Analysis
Through the generation of **WordClouds** and **N-Gram Frequency Distributions**, several key insights were obtained:
* The most frequent words in both positive and negative classes are highly similar (e.g., "movie", "film", "one", "like"). This indicates that simple frequency counting is insufficient for classification.
* Significant sentiment indicators only emerge when analyzing **Bigrams** (e.g., "worst movie", "highly recommend").

---

## 3. Dimensionality Reduction (Truncated SVD)

Text data is inherently high-dimensional. A standard TF-IDF (Term Frequency-Inverse Document Frequency) vectorization of our dataset yields a sparse matrix of 5,000 to 10,000 dimensions (one for each unique word).

To satisfy the **Dimensionality Reduction** requirement of the assignment, we applied **Truncated Singular Value Decomposition (SVD)**. Standard PCA (Principal Component Analysis) fails on sparse matrices because it requires centering the data (which destroys the sparsity and consumes massive RAM). Truncated SVD operates directly on sparse matrices.

* **Visualization:** We successfully compressed the 5,000-dimensional space down to **2 dimensions** (Principal Component 1 and 2), allowing us to plot the reviews on a standard X/Y scatter plot. While there is overlap due to extreme compression, distinct clustering of negative and positive vocabularies is visible.
* **Dimensionality Proof:** We reduced the dataset to 100 dimensions and trained a Logistic Regression model, proving that we can discard 98% of the feature space while retaining significant predictive accuracy.

---

## 4. Methodology: Machine Learning & Deep Learning Models

To comprehensively evaluate sentiment classification, we implemented four distinct models, scaling from traditional statistical methods to state-of-the-art Deep Learning.

### 4.1 Multinomial Naive Bayes (Baseline)
* **Feature Extraction:** TF-IDF Vectorization (Unigrams + Bigrams, max 10,000 features).
* **Architecture:** Based on Bayes' Theorem with the "naive" assumption of conditional independence between words. 
* **Results:** Achieved **84.20% Accuracy**.
* **Limitations:** The "bag-of-words" approach completely ignores word order. Therefore, it struggles heavily with negation ("not good") and sarcasm, treating it identically to "good".

### 4.2 Logistic Regression
* **Feature Extraction:** TF-IDF Vectorization.
* **Architecture:** An L2-regularized linear classifier designed to draw a hyperplane through the high-dimensional TF-IDF space.
* **Results:** Achieved **89.5% Accuracy**. It outperformed Naive Bayes by establishing better probabilistic weights for crucial sentiment-bearing words.

### 4.3 Bidirectional LSTM (Recurrent Neural Network)
* **Architecture:** Built from scratch using PyTorch. Instead of TF-IDF, it uses an **Embedding Layer** to map tokens to 128-dimensional dense vectors.
* **Bidirectionality:** Processes the sequence of words both forwards and backwards to capture historical and future context.
* **Results:** Achieved **87.78% Accuracy**. While slightly lower than the highly-optimized Logistic Regression, this model is theoretically superior because it learns structural grammar relationships purely from scratch.

### 4.4 BERT (Transformer)
* **Architecture:** Bidirectional Encoder Representations from Transformers (`bert-base-uncased`). 
* **Methodology:** Rather than training from scratch, we utilized **Transfer Learning**. BERT was pre-trained on the entirety of Wikipedia. We fine-tuned its classification head over 3 epochs on our IMDB dataset using a GPU.
* **Results:** Achieved an outstanding **94.16% Accuracy** and 0.9416 Macro F1-Score. By utilizing Self-Attention mechanisms, BERT perfectly grasps the context of every word relative to the entire sentence, overcoming the sarcasm and negation limitations of traditional models.

---

## 5. Key Findings and Conclusions

1. **Context is King:** The progression of accuracy directly correlates with a model's ability to understand context. Naive Bayes (84%) assumes words are independent. LSTMs (87%) understand immediate sequential history. BERT (94%) uses self-attention to understand the entire bidirectional context of a sentence simultaneously. 
2. **The Limit of Bag-of-Words:** Traditional ML (Logistic Regression / Naive Bayes) is exceptionally fast and highly interpretable, but they hit a hard accuracy ceiling (~89%) due to their inability to process complex sentence structures like sarcasm.
3. **Web Deployment:** To ensure these findings are easily accessible, the entire methodology, EDA visualizations, and model comparisons have been deployed to a responsive, multi-page website hosted on GitHub Pages (`/docs` folder).

This project successfully fulfills all Midterm deliverables, demonstrating end-to-end data science proficiency from raw HTML-ridden text to state-of-the-art Transformer predictions.
