import React, { useEffect, useState } from 'react';
import './index.css';

const App = () => {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <div className={`app-container ${mounted ? 'animate-in' : ''}`}>
      <div className="bg-glow"></div>
      <div className="bg-glow-right"></div>

      <main>
        {/* HERO SECTION */}
        <section className="container hero" style={{ minHeight: '80vh', display: 'flex', flexDirection: 'column', justifyContent: 'center', textAlign: 'center' }}>
          <h1 style={{ fontSize: '4rem', marginBottom: '16px' }}>Data <span className="gradient-text">Speaks!!</span></h1>
          <h2 style={{ fontSize: '2rem', fontWeight: 400, color: 'var(--text-secondary)', marginBottom: '32px' }}>
            Sentiment Analysis and Rating Prediction on IMDB
          </h2>
          <p style={{ fontSize: '1.2rem', maxWidth: '600px', margin: '0 auto', color: '#ccc' }}>
            A comprehensive machine learning pipeline spanning Binary Classification, Continuous Regression, and Unsupervised Clustering across 100,000 movie reviews.
          </p>
          <div style={{ marginTop: '48px', fontSize: '1rem', color: 'var(--accent-purple)', fontWeight: 'bold', letterSpacing: '2px' }}>
            BY TEAM 13
          </div>
        </section>

        {/* ABSTRACT SECTION */}
        <section className="container">
          <div className="glass-card">
            <h3 style={{ fontSize: '2rem', marginBottom: '24px', color: 'var(--accent-cyan)' }}>Abstract & Architecture</h3>
            <p>
              This project decodes the inherent complexity of human language by analyzing 100,000 IMDB reviews. We contrast traditional statistical methods (Naive Bayes, XGBoost) against modern Deep Learning architectures (LSTMs, BERT) to solve three distinct NLP paradigms: predicting sentiment polarity, forecasting exact 1-10 numerical ratings, and discovering latent semantic clusters without labels.
            </p>
          </div>
        </section>

        {/* EDA SECTION */}
        <section className="container">
          <h3 style={{ fontSize: '2.5rem', marginBottom: '40px', textAlign: 'center' }}>Exploratory Data <span className="gradient-text">Analysis</span></h3>
          <div className="grid-3">
            <div className="glass-card">
              <h4 style={{ fontSize: '1.5rem', marginBottom: '16px', color: '#fff' }}>N-Gram Importance</h4>
              <p>Analysis revealed that generic unigrams (stopwords) hold minimal predictive power. However, bigrams like "highly recommend" and "worst movie" act as critical discriminators for sentiment.</p>
            </div>
            <div className="glass-card">
              <h4 style={{ fontSize: '1.5rem', marginBottom: '16px', color: '#fff' }}>Punctuation Density</h4>
              <p>A strong correlation exists between punctuation usage (e.g., !!!) and extreme negative sentiment, providing a valuable secondary predictive signal for 1-star reviews.</p>
            </div>
            <div className="glass-card">
              <h4 style={{ fontSize: '1.5rem', marginBottom: '16px', color: '#fff' }}>Review Length</h4>
              <p>A prominent U-shaped distribution was observed: users write significantly longer reviews for highly polarized opinions (1-star and 10-star) compared to moderate ratings.</p>
            </div>
          </div>
        </section>

        {/* CLASSIFICATION SECTION */}
        <section className="container">
          <h3 style={{ fontSize: '2.5rem', marginBottom: '40px' }}>Binary <span className="gradient-text">Classification</span></h3>
          <div className="grid-2" style={{ alignItems: 'center' }}>
            <div>
              <p style={{ marginBottom: '24px' }}>
                We evaluated four models of increasing complexity to predict whether a review was positive or negative. The evolution from bag-of-words to bidirectional self-attention perfectly illustrates the necessity of contextual awareness in NLP.
              </p>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--glass-border)', paddingBottom: '8px' }}>
                  <span>Multinomial Naive Bayes</span> <strong style={{ color: 'var(--accent-cyan)' }}>84.20%</strong>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--glass-border)', paddingBottom: '8px' }}>
                  <span>Logistic Regression (TF-IDF)</span> <strong style={{ color: 'var(--accent-cyan)' }}>88.22%</strong>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--glass-border)', paddingBottom: '8px' }}>
                  <span>Bidirectional LSTM</span> <strong style={{ color: 'var(--accent-cyan)' }}>87.78%</strong>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--glass-border)', paddingBottom: '8px' }}>
                  <span style={{ color: 'var(--accent-purple)', fontWeight: 'bold' }}>Fine-Tuned BERT</span> <strong style={{ color: 'var(--accent-purple)', fontSize: '1.2rem' }}>94.16%</strong>
                </div>
              </div>
            </div>
            <div className="glass-card" style={{ padding: '16px' }}>
              <img src="/assets/bert_confusion_matrix.png" alt="BERT Confusion Matrix" style={{ width: '100%', borderRadius: '12px' }} />
              <p style={{ textAlign: 'center', marginTop: '16px', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                BERT Confusion Matrix demonstrating minimal misclassifications on the test set.
              </p>
            </div>
          </div>
        </section>

        {/* REGRESSION SECTION */}
        <section className="container">
          <h3 style={{ fontSize: '2.5rem', marginBottom: '40px', textAlign: 'right' }}>Continuous <span className="gradient-text">Regression</span></h3>
          <div className="grid-3" style={{ marginBottom: '32px' }}>
            <div className="glass-card" style={{ textAlign: 'center' }}>
              <div className="metric">2.16</div>
              <div className="metric-label">XGBoost MAE</div>
              <p style={{ marginTop: '16px', fontSize: '0.9rem' }}>Ensemble of trees on sparse TF-IDF features.</p>
            </div>
            <div className="glass-card" style={{ textAlign: 'center' }}>
              <div className="metric">1.54</div>
              <div className="metric-label">LSTM MAE</div>
              <p style={{ marginTop: '16px', fontSize: '0.9rem' }}>Sequential modeling with MSE loss.</p>
            </div>
            <div className="glass-card" style={{ textAlign: 'center', borderColor: 'rgba(138, 43, 226, 0.4)' }}>
              <div className="metric" style={{ color: 'var(--accent-purple)' }}>0.99</div>
              <div className="metric-label" style={{ color: '#fff' }}>BERT MAE</div>
              <p style={{ marginTop: '16px', fontSize: '0.9rem' }}>State-of-the-art contextual regression.</p>
            </div>
          </div>
          <div className="glass-card">
            <p>
              Predicting the exact 1-10 rating introduces extreme difficulty. Distinguishing a "7/10" review from an "8/10" review requires the model to understand subtle linguistic qualifiers. BERT achieved an astonishing Mean Absolute Error (MAE) of 0.9897 (R² = 0.8154), meaning its predictions deviate by less than one single rating point on average.
            </p>
          </div>
        </section>

        {/* CLUSTERING SECTION */}
        <section className="container">
          <h3 style={{ fontSize: '2.5rem', marginBottom: '40px' }}>Unsupervised <span className="gradient-text">Clustering</span></h3>
          <div className="grid-2" style={{ alignItems: 'center' }}>
            <div className="glass-card" style={{ padding: '16px' }}>
              <img src="/assets/svd_2d_scatter.png" alt="SVD 2D Scatter Plot" style={{ width: '100%', borderRadius: '12px' }} />
              <p style={{ textAlign: 'center', marginTop: '16px', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                Truncated SVD compressing 5,000+ dimensions into 2 Principal Components.
              </p>
            </div>
            <div>
              <h4 style={{ fontSize: '1.5rem', marginBottom: '16px', color: 'var(--accent-cyan)' }}>Latent Semantic Groupings</h4>
              <p style={{ marginBottom: '16px' }}>
                For the final paradigm, a completely unlabelled dataset of 50,000 reviews was analyzed using unsupervised techniques. We utilized two distinct <strong>K-Means Clustering</strong> approaches to iteratively group the TF-IDF feature space into hard clusters, minimizing spatial variance.
              </p>
              <p>
                Recognizing the limitations of rigid clustering, <strong>Gaussian Mixture Models (GMM)</strong> were applied as a probabilistic soft-clustering alternative, effectively capturing ambiguous reviews that contained overlapping, highly mixed sentiments.
              </p>
            </div>
          </div>
        </section>

        {/* CONCLUSION */}
        <section className="container" style={{ paddingBottom: '150px' }}>
          <div className="glass-card" style={{ textAlign: 'center' }}>
            <h3 style={{ fontSize: '2rem', marginBottom: '24px' }}>Project <span className="gradient-text">Conclusions</span></h3>
            <p style={{ maxWidth: '800px', margin: '0 auto' }}>
              This project successfully proves that the choice of architecture heavily dictates the limits of natural language understanding. While traditional statistical models provide strong baselines, bidirectional self-attention is mandatory for capturing complex linguistic context. Furthermore, analyzing <em>how</em> users write provides immense secondary predictive value alongside analyzing <em>what</em> they write.
            </p>
          </div>
        </section>
      </main>
    </div>
  );
};

export default App;
