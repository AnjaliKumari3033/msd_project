import React from 'react';

const Home = () => {
  return (
    <div className="container">
      <main>
        <h1>Data Speaks!! — IMDB Sentiment Analysis</h1>
        <p style={{fontSize: '1.1rem', maxWidth: '800px'}}>
          A machine learning project by Team 13, exploring how well different models can understand human opinions from movie reviews.
        </p>

        {/* ── Problem Statement ── */}
        <section>
          <h2>Problem Statement</h2>
          <p>
            When people write movie reviews, they express opinions using complex language — sarcasm, slang, double negatives, and mixed feelings. Our goal is to build machine learning models that can accurately understand this language and answer two questions:
          </p>
          <div className="grid grid-cols-2">
            <div className="card">
              <h4>🎯 Question 1: Is this review positive or negative?</h4>
              <p>
                Given a movie review, classify it as <strong>positive</strong> or <strong>negative</strong>. This is a <em>binary classification</em> task. For example, "This movie was a complete waste of time" → Negative.
              </p>
            </div>
            <div className="card">
              <h4>🎯 Question 2: What exact rating would this reviewer give?</h4>
              <p>
                Go beyond just positive/negative and predict the <strong>exact rating on a 1–10 scale</strong>. This is a <em>continuous regression</em> task — much harder because the model must distinguish between subtle differences like a 7/10 and an 8/10 review.
              </p>
            </div>
          </div>
          <p style={{marginTop: '20px'}}>
            Additionally, we explore whether <strong>unsupervised clustering</strong> can discover natural sentiment groupings in reviews without any labels at all — letting the data "speak for itself."
          </p>
        </section>

        {/* ── Dataset ── */}
        <section>
          <h2>Dataset</h2>
          <p>
            We use the <strong>IMDB Large Movie Review Dataset</strong>, one of the most widely-used benchmarks for sentiment analysis in NLP research. It was originally published by Stanford researchers (Maas et al., 2011).
          </p>

          <div className="stat-box">
            <div className="stat-label">Total Reviews</div>
            <div className="stat-value">100,000</div>
          </div>

          <p>
            The raw data comes as 100,000 individual <code>.txt</code> files organized in folders. We parsed these files into three separate datasets, one for each learning task:
          </p>

          <table className="model-table">
            <thead>
              <tr>
                <th>Dataset</th>
                <th>Size</th>
                <th>Labels</th>
                <th>Used For</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Binary Classification</strong></td>
                <td>50,000 reviews</td>
                <td>Positive (1) or Negative (0), from folder names</td>
                <td>Training classifiers (Naive Bayes → BERT)</td>
              </tr>
              <tr>
                <td><strong>Continuous Regression</strong></td>
                <td>50,000 reviews</td>
                <td>Exact 1–10 rating, extracted from filenames</td>
                <td>Predicting star ratings (XGBoost → BERT)</td>
              </tr>
              <tr>
                <td><strong>Unsupervised</strong></td>
                <td>50,000 reviews</td>
                <td>No labels at all</td>
                <td>Clustering &amp; topic discovery (K-Means, GMM)</td>
              </tr>
            </tbody>
          </table>
        </section>

        {/* ── Six Data Anomalies ── */}
        <section>
          <h2>Six Data Anomalies We Found &amp; Fixed</h2>
          <p>
            Real-world data is messy. Before feeding reviews into any model, we audited the entire dataset and discovered <strong>six key problems</strong> that needed to be fixed. Ignoring these would have hurt model accuracy or introduced bias.
          </p>

          <div className="grid grid-cols-2">
            <div className="card">
              <h4>1. HTML Tags Embedded in Text</h4>
              <p>
                Since reviews were scraped from the IMDB website, about <strong>58% of all reviews</strong> contained leftover HTML tags like <code>&lt;br /&gt;</code>. These are meaningless to sentiment analysis and were stripped using regex.
              </p>
            </div>

            <div className="card">
              <h4>2. Encoded HTML Entities</h4>
              <p>
                Characters like <code>&amp;amp;</code>, <code>&amp;quot;</code>, and <code>&amp;lt;</code> appeared instead of their actual symbols (<code>&amp;</code>, <code>"</code>, <code>&lt;</code>). We decoded these back to restore the original text.
              </p>
            </div>

            <div className="card">
              <h4>3. Duplicate Reviews</h4>
              <p>
                We found <strong>96 duplicates in the training set</strong> and <strong>199 in the test set</strong>. Keeping duplicates would let models memorize repeated examples instead of actually learning, so we removed them.
              </p>
            </div>

            <div className="card">
              <h4>4. Missing Neutral Ratings (5 and 6)</h4>
              <p>
                The dataset creators intentionally excluded reviews rated 5 or 6 out of 10. This creates a gap in the middle of the rating scale, producing a <strong>U-shaped distribution</strong> that our regression models had to account for.
              </p>
            </div>

            <div className="card">
              <h4>5. Inconsistent Whitespace &amp; Hidden Characters</h4>
              <p>
                After removing HTML tags, reviews were left with irregular spacing — double spaces, hidden newlines, and tab characters. We normalized all whitespace to single spaces so the same word isn't treated as two different tokens.
              </p>
            </div>

            <div className="card">
              <h4>6. Extreme Length Variation</h4>
              <p>
                Review lengths ranged from just <strong>32 characters to over 13,700 characters</strong>. This massive variation required careful handling — truncation for BERT's 512-token limit and padding for batch processing in LSTM.
              </p>
            </div>
          </div>
        </section>

        {/* ── Key EDA Insights ── */}
        <section>
          <h2>Key Findings from Data Exploration</h2>
          <p>
            After cleaning, we explored the data to understand patterns that could guide our model design. Here are the most important discoveries:
          </p>

          <div className="grid grid-cols-3">
            <div className="card">
              <h4>📊 N-Grams Matter More Than Single Words</h4>
              <p>
                Individual words like "good" or "bad" appear in both positive and negative reviews. But word pairs (bigrams) like <strong>"worst movie"</strong> or <strong>"highly recommend"</strong> are much more predictive of sentiment.
              </p>
            </div>

            <div className="card">
              <h4>😡 Angry Reviews Use More Punctuation</h4>
              <p>
                1-star reviews contain significantly more exclamation marks (<code>!!!</code>) and ALL-CAPS words than any other rating. Negative sentiment is expressed more aggressively than positive sentiment.
              </p>
            </div>

            <div className="card">
              <h4>📏 Extreme Opinions = Longer Reviews</h4>
              <p>
                Reviews with ratings of 1 or 10 are substantially longer than moderate reviews (4–7). People write more when they feel strongly — whether they love or hate the movie.
              </p>
            </div>
          </div>
        </section>

        {/* ── Quick Navigation ── */}
        <section>
          <h2>Explore Our Models</h2>
          <p>
            Use the navigation bar above to explore the three modeling tasks in detail:
          </p>
          <div className="grid grid-cols-3">
            <div className="stat-box">
              <div className="stat-label">Binary Classification</div>
              <div className="stat-value">94.16%</div>
              <p style={{margin: 0, fontSize: '0.85rem'}}>Best accuracy (BERT)</p>
            </div>
            <div className="stat-box">
              <div className="stat-label">Continuous Regression</div>
              <div className="stat-value">0.99 MAE</div>
              <p style={{margin: 0, fontSize: '0.85rem'}}>Predicts within ~1 star (BERT)</p>
            </div>
            <div className="stat-box">
              <div className="stat-label">Unsupervised Clustering</div>
              <div className="stat-value">K-Means + GMM</div>
              <p style={{margin: 0, fontSize: '0.85rem'}}>Natural sentiment groups discovered</p>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
};

export default Home;
