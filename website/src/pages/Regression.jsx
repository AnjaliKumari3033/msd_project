import React from 'react';

const Regression = () => {
  return (
    <div className="container">
      <main>
        <h1>Dataset 2: Continuous Rating Regression</h1>
        
        <section>
          <h2>1. Data Extraction & Origins</h2>
          <p>
            Unlike binary classification which only requires a broad positive/negative label, regression demands a precise continuous target variable. To build this dataset, we extracted the exact 1-10 numerical rating assigned by the reviewer.
          </p>
          <p>
            This data was embedded directly within the original raw filenames. For instance, a file named <code>1234_7.txt</code> indicates a review ID of 1234 and an exact user rating of 7/10. We developed a regular expression parser (<code>_(\d+)\.txt</code>) to extract these integers during the directory traversal. The final dataset consists of 50,000 reviews mapped directly to their explicit 1-10 rating, providing a much higher-fidelity target for our regression models.
          </p>
        </section>

        <section>
          <h2>2. Exploratory Data Analysis (EDA)</h2>
          <p>
            Understanding the distribution of the continuous 1-10 ratings is critical, as regression models can be highly sensitive to class imbalances or skewed targets.
          </p>
          
          <div className="grid grid-cols-2">
            <div className="card">
              <h3>Rating Distribution</h3>
              <p>
                Our analysis of the exact 1-10 labels revealed a distinct bimodal distribution. Reviewers are overwhelmingly more likely to leave extreme ratings (1s and 10s) rather than moderate ratings (4s, 5s, 6s). This polarization suggests that users only take the time to write a review when they have a very strong opinion about the film.
              </p>
              <img src="/assets/eda_plot_3.png" alt="Rating Distribution Histogram" />
              <div className="img-caption">Histogram showing the bimodal distribution of 1-10 ratings</div>
            </div>
            
            <div className="card">
              <h3>Punctuation Density vs Rating</h3>
              <p>
                We engineered a feature to track the density of exclamation marks and capital letters. A strong negative correlation was observed: reviews with a high density of consecutive punctuation (e.g., "!!!") were highly correlated with extreme negative ratings (1/10).
              </p>
              <img src="/assets/eda_plot_4.png" alt="Punctuation Analysis" />
              <div className="img-caption">Correlation between punctuation usage and average rating</div>
            </div>
          </div>
        </section>

        <section>
          <h2>3. Models & Statistics</h2>
          <p>
            Predicting an exact 1-10 numerical rating is a fundamentally harder problem than binary classification. The model must understand subtle linguistic qualifiers to distinguish between a "7/10" and an "8/10" review. We evaluated models based on Mean Absolute Error (MAE) and Mean Squared Error (MSE).
          </p>
          
          <table className="model-table">
            <thead>
              <tr>
                <th>Model Architecture</th>
                <th>Description</th>
                <th>MAE (Mean Absolute Error)</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>XGBoost Regressor</strong></td>
                <td>An optimized distributed gradient boosting library. Trees were trained on sparse TF-IDF vectors, acting as a strong non-linear statistical baseline.</td>
                <td><strong>2.16</strong></td>
              </tr>
              <tr>
                <td><strong>LSTM Regressor</strong></td>
                <td>The Bidirectional LSTM architecture adapted with a linear dense output layer and trained using Mean Squared Error (MSE) loss, capturing temporal phrasing.</td>
                <td><strong>1.54</strong></td>
              </tr>
              <tr>
                <td><strong>Fine-Tuned BERT</strong></td>
                <td>The transformer architecture fine-tuned specifically for regression by replacing the classification head with a linear layer and utilizing MSE loss.</td>
                <td><strong>0.99</strong></td>
              </tr>
            </tbody>
          </table>

          <div className="grid grid-cols-2" style={{marginTop: '30px'}}>
            <div>
              <h3>BERT Regression Performance</h3>
              <p>
                BERT achieved an astonishing Mean Absolute Error of 0.9897. This implies that, on average, the model's prediction deviates from the true human rating by less than one single point on the 1-10 scale. The R² score of 0.8154 confirms that the model explains a vast majority of the variance in the ratings.
              </p>
              <div className="grid grid-cols-2">
                <div className="stat-box">
                  <div className="stat-label">Lowest MAE</div>
                  <div className="stat-value">0.99</div>
                </div>
                <div className="stat-box" style={{borderLeftColor: 'var(--text-tertiary)'}}>
                  <div className="stat-label">Explained Variance (R²)</div>
                  <div className="stat-value">0.81</div>
                </div>
              </div>
            </div>
            <div>
              <img src="/assets/bert_loss_curve.png" alt="BERT Loss Curve" />
              <div className="img-caption">Training and Validation Loss (MSE) during BERT fine-tuning</div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
};

export default Regression;
