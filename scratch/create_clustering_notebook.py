import nbformat as nbf
import os

def create_clustering_notebook():
    nb = nbf.v4.new_notebook()

    cells = []

    # Markdown
    cells.append(nbf.v4.new_markdown_cell("""# 🌟 Unsupervised Learning: Topic Modeling & Clustering
In this notebook, we explore the **50,000 unlabelled reviews** from the IMDB dataset. 
Since these reviews do not have sentiment labels, we will use Unsupervised Learning—specifically **Latent Dirichlet Allocation (LDA)**—to automatically group the reviews into clusters or "Topics" based on their vocabulary.
"""))

    # Imports
    cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

sns.set_theme(style="whitegrid")
"""))

    # Load Data
    cells.append(nbf.v4.new_markdown_cell("""## 1. Load the Unsupervised Dataset"""))
    cells.append(nbf.v4.new_code_cell("""# Load the 50,000 unlabelled reviews
df = pd.read_csv("../processed_for_clustering/imdb_unsup.csv")

print(f"Dataset shape: {df.shape}")
df.head()
"""))

    # Clean Text
    cells.append(nbf.v4.new_markdown_cell("""## 2. Text Preprocessing
We will remove HTML tags from the reviews. We don't need to manually remove stop words or punctuation, as `TfidfVectorizer` can handle that natively.
"""))
    cells.append(nbf.v4.new_code_cell("""def clean_html(text):
    # Remove HTML tags like <br />
    text = re.sub(r'<[^>]*>', ' ', str(text))
    return text

df['clean_text'] = df['text'].apply(clean_html)
df[['text', 'clean_text']].head(3)
"""))

    # Vectorization
    cells.append(nbf.v4.new_markdown_cell("""## 3. TF-IDF Vectorization
We convert the text into a TF-IDF matrix. To prevent our machine from running out of memory on 50,000 documents, we will limit the vocabulary to the top **5,000** most significant words. We also remove common English stop words.
"""))
    cells.append(nbf.v4.new_code_cell("""tfidf = TfidfVectorizer(max_features=5000, stop_words='english', lowercase=True)

# Fit and transform the clean text
X_tfidf = tfidf.fit_transform(df['clean_text'])

print(f"TF-IDF Matrix shape: {X_tfidf.shape}")
"""))

    # LDA Modeling
    cells.append(nbf.v4.new_markdown_cell("""## 4. Latent Dirichlet Allocation (LDA) Topic Modeling
We will configure LDA to find **5 latent topics** across the corpus.
"""))
    cells.append(nbf.v4.new_code_cell("""n_topics = 5
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, n_jobs=-1)

# This might take a couple of minutes to run on 50,000 documents
lda.fit(X_tfidf)

print("LDA model fitted successfully!")
"""))

    # Extract Keywords
    cells.append(nbf.v4.new_markdown_cell("""## 5. Discovering the Topics
Let's see what words the LDA model has grouped together for each of the 5 topics.
"""))
    cells.append(nbf.v4.new_code_cell("""def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx + 1}:")
        # Get the indices of the top words
        top_indices = topic.argsort()[:-no_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_indices]
        print(", ".join(top_words))
        print("-" * 50)

feature_names = tfidf.get_feature_names_out()
display_topics(lda, feature_names, no_top_words=10)
"""))

    # Assign Topics
    cells.append(nbf.v4.new_markdown_cell("""## 6. Assigning Topics to Reviews
Now we will calculate the topic probabilities for every single review, and assign each review to the topic it has the highest probability of belonging to.
"""))
    cells.append(nbf.v4.new_code_cell("""# Transform the TF-IDF matrix into topic probabilities
topic_probabilities = lda.transform(X_tfidf)

# Find the dominant topic for each review (0 to 4)
dominant_topic = np.argmax(topic_probabilities, axis=1)

df['dominant_topic'] = dominant_topic + 1  # 1-indexed for readability

print(df['dominant_topic'].value_counts().sort_index())
"""))

    # Visualizing Topic Distribution
    cells.append(nbf.v4.new_markdown_cell("""## 7. Visualizing the Topic Distribution
Let's plot how the 50,000 reviews are distributed among the 5 discovered topics.
"""))
    cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='dominant_topic', palette='viridis', order=[1,2,3,4,5])
plt.title("Distribution of 50,000 Reviews Across 5 LDA Topics", fontsize=16)
plt.xlabel("Topic Number", fontsize=12)
plt.ylabel("Number of Reviews", fontsize=12)

# Create an artifacts folder to save the plot
import os
os.makedirs("artifacts", exist_ok=True)
plt.savefig("artifacts/topic_distribution.png", dpi=300, bbox_inches='tight')
plt.show()
"""))

    # Examine random reviews
    cells.append(nbf.v4.new_markdown_cell("""## 8. Examining Example Reviews
Let's print one sample review from each topic to see if the assigned topic matches the content!
"""))
    cells.append(nbf.v4.new_code_cell("""for i in range(1, 6):
    print(f"--- Example Review from Topic {i} ---")
    sample_text = df[df['dominant_topic'] == i].sample(1, random_state=42)['clean_text'].values[0]
    # Print the first 500 characters
    print(sample_text[:500] + "...")
    print("\\n")
"""))

    nb.cells = cells
    
    os.makedirs(r"c:\Users\dhanu\.gemini\antigravity-ide\scratch\msd_project\clustering", exist_ok=True)
    out_path = r"c:\Users\dhanu\.gemini\antigravity-ide\scratch\msd_project\clustering\CLUSTERING_UNSUP.ipynb"
    
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
        
    print(f"Created notebook at {out_path}")

if __name__ == "__main__":
    create_clustering_notebook()
