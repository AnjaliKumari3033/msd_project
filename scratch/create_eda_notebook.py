import nbformat as nbf
import os

def create_eda_notebook():
    nb = nbf.v4.new_notebook()

    cells = []

    # Markdown Intro
    cells.append(nbf.v4.new_markdown_cell("""# 📊 Comprehensive Exploratory Data Analysis (EDA)

This notebook explores the three distinct datasets in our project:
1. **Binary Classification Dataset** (25,000 labeled reviews: Positive/Negative)
2. **Continuous Regression Dataset** (25,000 labeled reviews: 1 to 10 Ratings)
3. **Unsupervised Dataset** (50,000 unlabelled reviews)

We will compare these datasets side-by-side to understand text length distributions, class balance, and natural sentiment in the wild!
"""))

    # Imports
    cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from wordcloud import WordCloud
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

# Set aesthetic parameters
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12})
"""))

    # Load Data
    cells.append(nbf.v4.new_markdown_cell("""## 1. Load All Datasets"""))
    cells.append(nbf.v4.new_code_cell("""print("Loading Classification Data...")
df_cls = pd.read_csv("../processed/imdb_train.csv")

print("Loading Regression Data...")
df_reg = pd.read_csv("../processed_for_regression/imdb_regression_train.csv")

print("Loading Unsupervised Data...")
df_unsup = pd.read_csv("../processed_for_clustering/imdb_unsup.csv")

print(f"\\nClassification: {df_cls.shape}")
print(f"Regression    : {df_reg.shape}")
print(f"Unsupervised  : {df_unsup.shape}")
"""))

    # Target Distribution
    cells.append(nbf.v4.new_markdown_cell("""## 2. Target Distributions (Classification vs. Regression)
Let's see how our target variables are distributed. We will plot the 50/50 binary split next to the 1-10 rating histogram.
"""))
    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Binary Classification
sns.countplot(data=df_cls, x='label', hue='label', legend=False, ax=axes[0], palette=['#e74c3c', '#2ecc71'])
axes[0].set_title("Binary Classification Balance (0=Neg, 1=Pos)", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Sentiment Label")
axes[0].set_ylabel("Count")

# Subplot 2: Regression Rating Distribution
sns.countplot(data=df_reg, x='rating', hue='rating', legend=False, ax=axes[1], palette="viridis")
axes[1].set_title("Regression Rating Distribution (1-10)", fontsize=14, fontweight='bold')
axes[1].set_xlabel("Rating Score")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.show()
"""))
    
    # Text Length Analysis
    cells.append(nbf.v4.new_markdown_cell("""## 3. Review Length Analysis (Across All 3 Datasets)
Are unlabelled reviews longer? Do people write longer reviews when they hate a movie (1) vs when they love it (10)? Let's compare!
"""))
    cells.append(nbf.v4.new_code_cell("""# Calculate string lengths
df_cls['text_length'] = df_cls['text'].str.len()
df_reg['text_length'] = df_reg['text'].str.len()
df_unsup['text_length'] = df_unsup['text'].str.len()

fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=False)

# 1. Length by Binary Class
sns.boxplot(data=df_cls, x='label', y='text_length', hue='label', legend=False, ax=axes[0], palette=['#e74c3c', '#2ecc71'])
axes[0].set_title("Text Length by Binary Sentiment", fontsize=14)
axes[0].set_ylim(0, 6000) # Cap at 6000 for visibility

# 2. Length by Rating
sns.boxplot(data=df_reg, x='rating', y='text_length', hue='rating', legend=False, ax=axes[1], palette="viridis")
axes[1].set_title("Text Length by Rating Score", fontsize=14)

# 3. Length Distribution (Unsupervised vs All)
sns.kdeplot(df_cls['text_length'], label='Classification', fill=True, alpha=0.3, ax=axes[2])
sns.kdeplot(df_unsup['text_length'], label='Unsupervised', fill=True, alpha=0.3, ax=axes[2])
axes[2].set_title("Overall Length Distribution", fontsize=14)
axes[2].legend()

plt.tight_layout()
plt.show()
"""))

    # Word Clouds
    cells.append(nbf.v4.new_markdown_cell("""## 4. Word Clouds (Positive vs. Negative)
Let's visually extract the most dominant words for Positive and Negative sentiments from the classification dataset.
"""))
    cells.append(nbf.v4.new_code_cell("""# Basic text cleaning for wordcloud
def clean_for_wc(text):
    text = re.sub(r'<[^>]*>', ' ', str(text))
    text = re.sub(r'[^a-zA-Z\\s]', '', text)
    return text.lower()

pos_text = " ".join(df_cls[df_cls['label'] == 1]['text'].sample(2000, random_state=42).apply(clean_for_wc))
neg_text = " ".join(df_cls[df_cls['label'] == 0]['text'].sample(2000, random_state=42).apply(clean_for_wc))

wc_pos = WordCloud(width=600, height=400, background_color='white', colormap='Greens', max_words=100).generate(pos_text)
wc_neg = WordCloud(width=600, height=400, background_color='black', colormap='Reds', max_words=100).generate(neg_text)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].imshow(wc_pos, interpolation='bilinear')
axes[0].set_title("Positive Reviews", fontsize=16, fontweight='bold')
axes[0].axis("off")

axes[1].imshow(wc_neg, interpolation='bilinear')
axes[1].set_title("Negative Reviews", fontsize=16, fontweight='bold', color='red')
axes[1].axis("off")

plt.tight_layout()
plt.show()
"""))

    # High Impact Words
    cells.append(nbf.v4.new_markdown_cell("""## 5. High-Impact Words (Regression Score Mapping)
What is the average 1-10 rating of a review if it contains the word "masterpiece"? What if it contains "awful"? 
Let's map specific vocabulary to their continuous regression score!
"""))
    cells.append(nbf.v4.new_code_cell("""keywords = ['masterpiece', 'excellent', 'good', 'average', 'boring', 'awful', 'worst']
avg_ratings = []

df_reg['clean_text'] = df_reg['text'].str.lower()

for word in keywords:
    # Find rows containing the word
    mask = df_reg['clean_text'].str.contains(r'\\b' + word + r'\\b', na=False)
    avg_rating = df_reg[mask]['rating'].mean()
    avg_ratings.append(avg_rating)

plt.figure(figsize=(10, 6))
sns.barplot(x=keywords, y=avg_ratings, hue=keywords, legend=False, palette="coolwarm_r")
plt.axhline(5.5, color='black', linestyle='--', label='Neutral Rating (5.5)')
plt.title("Average Rating of Reviews Containing Specific Words", fontsize=16, fontweight='bold')
plt.ylabel("Average 1-10 Rating")
plt.ylim(1, 10)
plt.legend()
plt.show()
"""))

    # Unsupervised Sentiment
    cells.append(nbf.v4.new_markdown_cell("""## 6. Sentiment "In the Wild" (Unsupervised Dataset)
Since the unsupervised dataset has no labels, we will use VADER (a pre-trained lexicon) to predict sentiment on a sample of 5,000 reviews. 
This will tell us if natural, unlabelled movie reviews lean Positive or Negative.
"""))
    cells.append(nbf.v4.new_code_cell("""sia = SentimentIntensityAnalyzer()

# Sample 5,000 reviews to save time
sample_unsup = df_unsup.sample(5000, random_state=42).copy()

def get_vader_compound(text):
    # Quick strip of HTML
    text = re.sub(r'<[^>]*>', ' ', str(text))
    return sia.polarity_scores(text)['compound']

print("Calculating VADER sentiment scores...")
sample_unsup['vader_score'] = sample_unsup['text'].apply(get_vader_compound)

plt.figure(figsize=(10, 6))
sns.histplot(sample_unsup['vader_score'], bins=30, kde=True, color='purple')
plt.title("Distribution of Sentiment (VADER Compound Score) in Unlabelled Data", fontsize=16, fontweight='bold')
plt.xlabel("Sentiment Score (-1.0 Negative, +1.0 Positive)")
plt.ylabel("Number of Reviews")
plt.axvline(0, color='black', linestyle='--')
plt.show()

# Proportion
pos_prop = (sample_unsup['vader_score'] > 0).mean()
print(f"Percentage of Positively-leaning reviews in Unsupervised Data: {pos_prop:.1%}")
"""))

    nb.cells = cells
    
    out_dir = r"c:\Users\dhanu\.gemini\antigravity-ide\scratch\msd_project\notebooks"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "4_COMPREHENSIVE_EDA.ipynb")
    
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
        
    print(f"Created notebook at {out_path}")

if __name__ == "__main__":
    create_eda_notebook()
