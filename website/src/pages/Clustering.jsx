import React from 'react';

const Clustering = () => {
  return (
    <div className="container">
      <main>
        <h1>Dataset 3: Unsupervised Clustering</h1>
        
        <section>
          <h2>1. Data Extraction & Origins</h2>
          <p>
            The final paradigm of our project aimed to discover latent semantic structures within the text without relying on human-annotated labels. To simulate a true unsupervised environment, we stripped all 50,000 reviews of their folder-based labels and numerical ratings.
          </p>
          <p>
            The dataset was aggregated into a single, massive CSV file containing only the raw text of the reviews. By deliberately blinding the models to the ground-truth sentiment, we forced the clustering algorithms to mathematically group the reviews based entirely on inherent linguistic similarities, vocabulary overlap, and TF-IDF vector proximity.
          </p>
        </section>

        <section>
          <h2>2. Exploratory Data Analysis (EDA) & Dimensionality Reduction</h2>
          <p>
            Analyzing unstructured text requires projecting massive high-dimensional spaces into human-readable formats.
          </p>
          
          <div className="grid grid-cols-2">
            <div className="card">
              <h3>TF-IDF Vector Sparsity</h3>
              <p>
                Converting 50,000 reviews into a Bag-of-Words matrix resulted in a vocabulary size exceeding 80,000 unique words. This created a highly sparse matrix (over 99% zeros). Visualizing the document-term frequency confirmed that most reviews only utilize a tiny fraction of the total vocabulary, necessitating advanced dimensionality reduction.
              </p>
              <img src="/assets/eda_plot_5.png" alt="Sparsity Plot" />
              <div className="img-caption">Heatmap visualization of document-term matrix sparsity</div>
            </div>
            
            <div className="card">
              <h3>Truncated SVD (LSA)</h3>
              <p>
                To combat the curse of dimensionality before clustering, we applied Truncated Singular Value Decomposition (SVD), specifically Latent Semantic Analysis (LSA). By compressing the 80,000+ dimensions down to a 100-component latent space, we preserved 85% of the variance while eliminating noise and generic stopwords.
              </p>
              <img src="/assets/svd_2d_scatter.png" alt="SVD 2D Scatter Plot" />
              <div className="img-caption">2D projection of the latent semantic space post-SVD</div>
            </div>
          </div>
        </section>

        <section>
          <h2>3. Models & Statistics</h2>
          <p>
            With the text projected into a dense, continuous latent space, we applied two distinct unsupervised clustering algorithms to discover natural groupings.
          </p>
          
          <table className="model-table">
            <thead>
              <tr>
                <th>Model Architecture</th>
                <th>Description</th>
                <th>Key Statistical Metric</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>K-Means Clustering</strong></td>
                <td>A rigid, hard-clustering algorithm that partitions the data into K distinct clusters by minimizing spatial variance (inertia) within the SVD-reduced feature space.</td>
                <td><strong>Silhouette: 0.14</strong></td>
              </tr>
              <tr>
                <td><strong>Gaussian Mixture Models (GMM)</strong></td>
                <td>A probabilistic model assuming all data points are generated from a mixture of finite Gaussian distributions. It provides "soft clustering", calculating the exact probability a review belongs to each cluster.</td>
                <td><strong>Log-Likelihood Maximized</strong></td>
              </tr>
            </tbody>
          </table>

          <div className="grid grid-cols-2" style={{marginTop: '30px'}}>
            <div>
              <h3>Latent Grouping Observations</h3>
              <p>
                While K-Means successfully separated highly distinct reviews (e.g., horror vs. romance vocabularies), it struggled with nuanced, mixed-sentiment reviews. GMM proved vastly superior for this NLP task.
              </p>
              <p>
                Because GMM applies <em>soft clustering</em>, it correctly identified that many reviews exhibit overlapping distributions—for example, a review praising the acting but criticizing the directing mathematically belongs 60% to a "positive" cluster and 40% to a "negative" cluster. The probabilistic approach of GMM perfectly captured the inherent ambiguity of human language.
              </p>
            </div>
            <div className="grid grid-cols-1">
              <div className="stat-box">
                <div className="stat-label">Optimal K Value (Elbow Method)</div>
                <div className="stat-value">K = 2</div>
              </div>
              <div className="stat-box" style={{borderLeftColor: 'var(--text-tertiary)'}}>
                <div className="stat-label">Primary Axis of Variance</div>
                <div className="stat-value">Sentiment Polarity</div>
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
};

export default Clustering;
