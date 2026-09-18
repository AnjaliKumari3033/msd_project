import React from 'react';

const Classification = () => {
  return (
    <div className="container">
      <main>
        <h1>Dataset 1: Binary Sentiment Classification</h1>
        
        <section>
          <h2>1. Data Extraction & Origins</h2>
          <p>
            The raw IMDB dataset originated as 100,000 individual text files nested deeply within a folder hierarchy (<code>train/pos</code>, <code>train/neg</code>). To construct a structured dataset for binary classification, we developed a Python pipeline to recursively traverse these directories, read the contents of each `.txt` file into memory, and assign a binary label (1 for positive, 0 for negative) based on the parent folder name. 
          </p>
          <p>
            This process yielded a perfectly balanced dataset of 50,000 labeled reviews (25,000 positive, 25,000 negative), which was exported to a single CSV file for streamlined model ingestion.
          </p>
        </section>

        <section>
          <h2>2. Exploratory Data Analysis (EDA)</h2>
          <p>
            During our exploratory phase, we analyzed the textual properties of the reviews to understand underlying patterns that could aid our models.
          </p>
          
          <div className="grid grid-cols-2">
            <div className="card">
              <h3>Top N-Grams</h3>
              <p>
                Initial frequency analysis of unigrams revealed that generic stopwords dominated the corpus, offering little predictive value. However, examining bigrams and trigrams (e.g., "highly recommend", "worst movie ever") revealed strong discriminative phrases that clearly separate positive from negative sentiments.
              </p>
              <img src="/assets/eda_plot_1.png" alt="Top N-Grams Distribution" />
              <div className="img-caption">Distribution of the most frequent predictive N-Grams</div>
            </div>
            
            <div className="card">
              <h3>Review Length & Word Count</h3>
              <p>
                We analyzed the distribution of review lengths. Interestingly, users tend to write significantly longer reviews when expressing highly polarized opinions (either extremely positive or extremely negative) compared to moderate sentiments.
              </p>
              <img src="/assets/eda_plot_2.png" alt="Review Length Distribution" />
              <div className="img-caption">Density plot of word counts across sentiment classes</div>
            </div>
          </div>
        </section>

        <section>
          <h2>3. Models & Statistics</h2>
          <p>
            We trained four classification models of increasing complexity to predict sentiment polarity. The transition from simple statistical models to deep contextual transformers highlights the importance of word order in NLP.
          </p>
          
          <table className="model-table">
            <thead>
              <tr>
                <th>Model Architecture</th>
                <th>Description</th>
                <th>Test Accuracy</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Multinomial Naive Bayes</strong></td>
                <td>A probabilistic classifier based on applying Bayes' theorem with strong independence assumptions between features. Used with Bag-of-Words vectors.</td>
                <td><strong>84.20%</strong></td>
              </tr>
              <tr>
                <td><strong>Logistic Regression</strong></td>
                <td>A linear model utilizing a logistic function to model a binary dependent variable. Trained on TF-IDF weighted features to reduce the impact of generic frequent words.</td>
                <td><strong>88.22%</strong></td>
              </tr>
              <tr>
                <td><strong>Bidirectional LSTM</strong></td>
                <td>A Recurrent Neural Network architecture that processes sequences in both forward and backward directions, capturing temporal dependencies and word order.</td>
                <td><strong>87.78%</strong></td>
              </tr>
              <tr>
                <td><strong>Fine-Tuned BERT</strong></td>
                <td>A transformer-based machine learning technique for NLP pre-training. We fine-tuned the pre-trained `bert-base-uncased` model, utilizing bidirectional self-attention to capture deep linguistic context.</td>
                <td><strong>94.16%</strong></td>
              </tr>
            </tbody>
          </table>

          <div className="grid grid-cols-2" style={{marginTop: '30px'}}>
            <div>
              <h3>BERT Performance Analysis</h3>
              <p>
                BERT significantly outperformed all traditional statistical methods. The confusion matrix on the right demonstrates its robustness, showing minimal false positives and false negatives on the test set. 
              </p>
              <div className="stat-box">
                <div className="stat-label">Peak Model Accuracy</div>
                <div className="stat-value">94.16%</div>
              </div>
            </div>
            <div>
              <img src="/assets/bert_confusion_matrix.png" alt="BERT Confusion Matrix" />
            </div>
          </div>
        </section>
      </main>
    </div>
  );
};

export default Classification;
