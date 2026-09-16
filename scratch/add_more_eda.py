import nbformat as nbf
import os

def append_eda_cells():
    notebook_path = r"c:\Users\dhanu\.gemini\antigravity-ide\scratch\msd_project\notebooks\4_COMPREHENSIVE_EDA.ipynb"
    
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)

    # ---------------------------------------------------------
    # NEW FEATURE 1: Bigram Analysis
    # ---------------------------------------------------------
    nb.cells.append(nbf.v4.new_markdown_cell("""## 7. N-Gram Analysis (Top Bigrams)
Single words are great, but two-word phrases (Bigrams) often carry much more context (e.g., "waste of time" vs "highly recommend"). 
Let's look at the top 10 bigrams for Positive and Negative reviews!
"""))

    nb.cells.append(nbf.v4.new_code_cell("""from sklearn.feature_extraction.text import CountVectorizer

def get_top_bigrams(corpus, n=10):
    vec = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]

# Sample 5000 reviews from each class for speed
pos_corpus = df_cls[df_cls['label'] == 1]['text'].sample(5000, random_state=42).apply(clean_for_wc)
neg_corpus = df_cls[df_cls['label'] == 0]['text'].sample(5000, random_state=42).apply(clean_for_wc)

print("Calculating Positive Bigrams...")
top_pos_bigrams = get_top_bigrams(pos_corpus)
print("Calculating Negative Bigrams...")
top_neg_bigrams = get_top_bigrams(neg_corpus)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.barplot(x=[val[1] for val in top_pos_bigrams], y=[val[0] for val in top_pos_bigrams], 
            ax=axes[0], hue=[val[0] for val in top_pos_bigrams], legend=False, palette='Greens_r')
axes[0].set_title("Top 10 Positive Bigrams", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Frequency")

sns.barplot(x=[val[1] for val in top_neg_bigrams], y=[val[0] for val in top_neg_bigrams], 
            ax=axes[1], hue=[val[0] for val in top_neg_bigrams], legend=False, palette='Reds_r')
axes[1].set_title("Top 10 Negative Bigrams", fontsize=14, fontweight='bold')
axes[1].set_xlabel("Frequency")

plt.tight_layout()
plt.show()
"""))

    # ---------------------------------------------------------
    # NEW FEATURE 2: Meta-Features (Exclamation Marks & ALL CAPS)
    # ---------------------------------------------------------
    nb.cells.append(nbf.v4.new_markdown_cell("""## 8. Meta-Features Analysis (Do angry reviewers shout more?)
Let's see if there is a correlation between the 1-10 rating score and how people write. Do 1-star reviews use more exclamation marks (`!!!`) or ALL CAPS words compared to 10-star reviews?
"""))

    nb.cells.append(nbf.v4.new_code_cell("""# Count exclamation marks
df_reg['exclamation_count'] = df_reg['text'].apply(lambda x: x.count('!'))

# Count ALL CAPS words (words with > 2 letters that are entirely uppercase)
df_reg['caps_count'] = df_reg['text'].apply(lambda x: len(re.findall(r'\\b[A-Z]{2,}\\b', x)))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Exclamation Marks vs Rating
sns.barplot(data=df_reg, x='rating', y='exclamation_count', ax=axes[0], hue='rating', legend=False, palette='viridis')
axes[0].set_title("Average Exclamation Marks (!) per Review by Rating", fontsize=14, fontweight='bold')
axes[0].set_xlabel("1-10 Rating")
axes[0].set_ylabel("Avg Exclamation Marks")

# ALL CAPS vs Rating
sns.barplot(data=df_reg, x='rating', y='caps_count', ax=axes[1], hue='rating', legend=False, palette='magma')
axes[1].set_title("Average ALL CAPS Words per Review by Rating", fontsize=14, fontweight='bold')
axes[1].set_xlabel("1-10 Rating")
axes[1].set_ylabel("Avg ALL CAPS Words")

plt.tight_layout()
plt.show()
"""))

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
        
    print(f"Successfully appended new analysis cells to {notebook_path}")

if __name__ == "__main__":
    append_eda_cells()
