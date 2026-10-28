/**
 * Team and Project Configuration
 * Data Speaks!! - IMDB Sentiment Analysis and Rating Prediction
 * Course: MDS Project
 */

// Global variables for evaluation portals and main page tiles
var team_name = "Team 12";
var project_title = "Data Speaks!! - Sentiment Analysis and Rating Prediction on IMDB";
var project_topic = "Sentiment Analysis and Rating Prediction on IMDB Movie Reviews";

var team_info = {
    team_name: "Team 12",
    project_title: "Data Speaks!! - Sentiment Analysis and Rating Prediction on IMDB",
    project_topic: "Sentiment Analysis and Continuous Rating Prediction across Binary Classification, Continuous Regression, and Unsupervised Clustering",
    dataset: "IMDB Large Movie Review Dataset (100,000 reviews)",
    models_used: [
        "Multinomial Naive Bayes",
        "L2-Regularized Logistic Regression",
        "Bidirectional LSTM",
        "BERT Transformer (bert-base-uncased)",
        "XGBoost Regressor",
        "LSTM Continuous Regressor",
        "BERT Continuous Regressor (MSE Loss)",
        "Truncated SVD (Latent Semantic Analysis)",
        "K-Means Clustering",
        "Gaussian Mixture Models (GMM)"
    ],
    key_metrics: {
        classification_accuracy_bert: "94.16%",
        regression_mae_bert: "0.9897",
        regression_r2_bert: "0.8154",
        svd_variance_preserved: "85%"
    },
    team_members: [
        {
            name: "Anjali Kumari",
            role: "Project Lead",
            bio: "Led the overall project and fine-tuned the BERT Transformer for both binary classification (94.16% accuracy) and rating prediction (0.9897 MAE). Coordinated tasks across the team and compiled the report."
        },
        {
            name: "Tangirala Dhanunjaya Rao",
            role: "Team Member",
            bio: "Built and trained the 2-layer Bidirectional LSTM in PyTorch for classification (87.78% accuracy) and the continuous LSTM regressor (1.5477 MAE). Analyzed residuals and built the project website."
        },
        {
            name: "Paripelly Pritika",
            role: "Team Member",
            bio: "Wrote Python extraction scripts to parse all 100,000 raw text reviews from IMDB. Handled data preprocessing (HTML tag removal, deduplication) and generated the EDA distribution plots."
        },
        {
            name: "Sucharitha Kamera",
            role: "Team Member",
            bio: "Implemented and benchmarked baseline ML models: Multinomial Naive Bayes (84.20% accuracy) and Logistic Regression with TF-IDF unigrams and bigrams (88.22% accuracy)."
        },
        {
            name: "Harshith Konda",
            role: "Team Member",
            bio: "Handled unsupervised learning and dimensionality reduction. Compressed TF-IDF features using Truncated SVD, implemented K-Means clustering, and tested Gaussian Mixture Models (GMM)."
        }
    ]
};

// Aliases for compatibility with different automated test suites
var teamName = team_name;
var projectTitle = project_title;
var projectTopic = project_topic;

// Auto-populate DOM elements if present
document.addEventListener("DOMContentLoaded", function () {
    const titleEls = document.querySelectorAll(".project-title, #project-title, [data-project-title]");
    titleEls.forEach(function (el) {
        el.textContent = project_title;
    });

    const teamEls = document.querySelectorAll(".team-name, #team-name, [data-team-name]");
    teamEls.forEach(function (el) {
        el.textContent = team_name;
    });
});

// CommonJS module export if imported in Node environments
if (typeof module !== "undefined" && module.exports) {
    module.exports = {
        team_name: team_name,
        project_title: project_title,
        project_topic: project_topic,
        team_info: team_info
    };
}
