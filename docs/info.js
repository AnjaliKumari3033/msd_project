/**
 * Team and Project Configuration
 * Data Speaks!! - IMDB Sentiment Analysis and Rating Prediction
 * Course: MDS Project
 */

// Global variables for evaluation portals and main page tiles
var team_name = "Team 13";
var project_title = "Data Speaks!! - Sentiment Analysis and Rating Prediction on IMDB";
var project_topic = "Sentiment Analysis and Rating Prediction on IMDB Movie Reviews";

var team_info = {
    team_name: "Team 13",
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
            bio: "Led overall project orchestration and spearheaded BERT transformer fine-tuning for both binary classification (94.16% accuracy) and continuous regression (0.9897 MAE)."
        },
        {
            name: "Paripelly Pritika",
            role: "Team Member",
            bio: "Architected the automated data extraction pipelines parsing 100,000 raw text files, regex sanitization, deduplication, and comparative EDA distribution analysis."
        },
        {
            name: "Sucharitha Kamera",
            role: "Team Member",
            bio: "Developed baseline statistical classifiers including Multinomial Naive Bayes and engineered TF-IDF n-gram vectorization pipelines with L2-regularized Logistic Regression."
        },
        {
            name: "Harshith Konda",
            role: "Team Member",
            bio: "Implemented Truncated SVD dimensionality reduction, K-Means clustering optimization via the Elbow method, and Gaussian Mixture Models (GMM) for soft-clustering mixed reviews."
        },
        {
            name: "Tangirala Dhanunjaya Rao",
            role: "Team Member",
            bio: "Designed and trained the PyTorch Bidirectional LSTM neural networks, performed continuous regression residual analysis, and built the responsive web application dashboard."
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
