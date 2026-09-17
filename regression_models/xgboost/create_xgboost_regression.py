from pathlib import Path
import re
import html

import numpy as np
import pandas as pd
import nbformat as nbf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

project = Path('/workspaces/msd_project')
source_dir = project / 'processed_for_regression'
model_dir = project / 'regression_models' / 'xgboost'
artifacts_dir = model_dir / 'artifacts'
metrics_dir = artifacts_dir / 'metrics'
notebook_path = model_dir / 'xgboost_regression.ipynb'

for d in [model_dir, artifacts_dir, metrics_dir]:
    d.mkdir(parents=True, exist_ok=True)


def clean_text(text):
    text = html.unescape(str(text)) if pd.notna(text) else ''
    text = re.sub(r'<br\s*/?>', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

for filename in ['imdb_regression_train.csv', 'imdb_regression_test.csv']:
    df = pd.read_csv(source_dir / filename)
    if {'text', 'rating'} - set(df.columns):
        raise ValueError(f'{filename} is missing required columns')

    df['text_clean'] = df['text'].apply(clean_text)
    df = df.drop_duplicates(subset=['text_clean'], keep='first').copy()
    df = df[['text_clean', 'rating']].rename(columns={'text_clean': 'text'})

    out_name = 'xgboost_regression_train_clean.csv' if 'train' in filename else 'xgboost_regression_test_clean.csv'
    df.to_csv(artifacts_dir / out_name, index=False)

train_df = pd.read_csv(artifacts_dir / 'xgboost_regression_train_clean.csv')
test_df = pd.read_csv(artifacts_dir / 'xgboost_regression_test_clean.csv')

X_train, X_val, y_train, y_val = train_test_split(
    train_df['text'].astype(str),
    train_df['rating'].astype(float),
    test_size=0.10,
    random_state=42,
    stratify=np.round(train_df['rating']).astype(int),
)

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2,
    max_features=5000,
    strip_accents='unicode',
)

X_tr = vectorizer.fit_transform(X_train)
X_va = vectorizer.transform(X_val)
X_te = vectorizer.transform(test_df['text'].astype(str))

model = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=250,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='mae',
)
model.fit(X_tr, y_train)

val_pred = model.predict(X_va)
test_pred = model.predict(X_te)

metrics = pd.DataFrame({
    'set': ['validation', 'test'],
    'mae': [
        mean_absolute_error(y_val, val_pred),
        mean_absolute_error(test_df['rating'].astype(float), test_pred),
    ],
    'rmse': [
        np.sqrt(mean_squared_error(y_val, val_pred)),
        np.sqrt(mean_squared_error(test_df['rating'].astype(float), test_pred)),
    ],
    'r2': [
        r2_score(y_val, val_pred),
        r2_score(test_df['rating'].astype(float), test_pred),
    ],
})
metrics.to_csv(metrics_dir / 'xgboost_regression_metrics.csv', index=False)

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell(
        '# XGBoost Regression for IMDb Ratings\n\n'
        'This notebook reads the regression data from the `processed_for_regression` folder, cleans the text, converts it to TF-IDF features, trains an XGBoost regressor, and saves cleaned CSVs under `artifacts/`.\n\n'
        '### Is preprocessing required?\n'
        'Yes. XGBoost cannot train on raw text directly, so the review text must be cleaned and converted to numeric TF-IDF features before fitting the model.'
    ),
    nbf.v4.new_code_cell(
        "import os\n"
        "import re\n"
        "import html\n\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "from sklearn.feature_extraction.text import TfidfVectorizer\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n"
        "from xgboost import XGBRegressor\n\n"
        "PROJECT_ROOT = os.path.abspath('../..')\n"
        "DATA_DIR = os.path.join(PROJECT_ROOT, 'processed_for_regression')\n"
        "ARTIFACTS_DIR = os.path.join(os.getcwd(), 'artifacts')\n"
        "os.makedirs(ARTIFACTS_DIR, exist_ok=True)\n\n"
        "train_df = pd.read_csv(os.path.join(DATA_DIR, 'imdb_regression_train.csv'))\n"
        "test_df = pd.read_csv(os.path.join(DATA_DIR, 'imdb_regression_test.csv'))\n\n"
        "print(f'Train shape: {train_df.shape}')\n"
        "print(f'Test shape: {test_df.shape}')\n"
        "print(train_df.head(2).to_string(index=False))"
    ),
    nbf.v4.new_markdown_cell(
        '\n---\n\n## Step 1: Clean the review text\n\n'
        'We remove HTML tags, decode HTML entities, normalize whitespace, and drop duplicate records before training.'
    ),
    nbf.v4.new_code_cell(
        "def clean_text(text):\n"
        "    text = html.unescape(str(text)) if pd.notna(text) else ''\n"
        "    text = re.sub(r'<br\\s*/?>', ' ', text, flags=re.IGNORECASE)\n"
        "    text = re.sub(r'<[^>]+>', ' ', text)\n"
        "    text = re.sub(r'\\s+', ' ', text).strip()\n"
        "    return text\n\n"
        "train_df['text_clean'] = train_df['text'].apply(clean_text)\n"
        "test_df['text_clean'] = test_df['text'].apply(clean_text)\n\n"
        "train_clean = train_df.drop_duplicates(subset=['text_clean'], keep='first').copy()\n"
        "test_clean = test_df.drop_duplicates(subset=['text_clean'], keep='first').copy()\n\n"
        "train_clean = train_clean[['text_clean', 'rating']].rename(columns={'text_clean': 'text'})\n"
        "test_clean = test_clean[['text_clean', 'rating']].rename(columns={'text_clean': 'text'})\n\n"
        "train_clean.to_csv(os.path.join(ARTIFACTS_DIR, 'xgboost_regression_train_clean.csv'), index=False)\n"
        "test_clean.to_csv(os.path.join(ARTIFACTS_DIR, 'xgboost_regression_test_clean.csv'), index=False)\n\n"
        "print(f'Train rows after cleaning: {len(train_clean)}')\n"
        "print(f'Test rows after cleaning: {len(test_clean)}')\n"
        "print(f'Saved cleaned data to: {ARTIFACTS_DIR}')"
    ),
    nbf.v4.new_markdown_cell(
        '\n---\n\n## Step 2: TF-IDF features and XGBoost training\n\n'
        'The cleaned text is converted into numeric vectors, which is the standard approach when training XGBoost on text data.'
    ),
    nbf.v4.new_code_cell(
        "train_clean = pd.read_csv(os.path.join(ARTIFACTS_DIR, 'xgboost_regression_train_clean.csv'))\n"
        "test_clean = pd.read_csv(os.path.join(ARTIFACTS_DIR, 'xgboost_regression_test_clean.csv'))\n\n"
        "X_train, X_val, y_train, y_val = train_test_split(\n"
        "    train_clean['text'].astype(str),\n"
        "    train_clean['rating'].astype(float),\n"
        "    test_size=0.10,\n"
        "    random_state=42,\n"
        "    stratify=np.round(train_clean['rating']).astype(int),\n"
        ")\n\n"
        "vectorizer = TfidfVectorizer(\n"
        "    lowercase=True,\n"
        "    stop_words='english',\n"
        "    ngram_range=(1, 2),\n"
        "    min_df=2,\n"
        "    max_features=5000,\n"
        "    strip_accents='unicode',\n"
        ")\n\n"
        "X_tr = vectorizer.fit_transform(X_train)\n"
        "X_va = vectorizer.transform(X_val)\n"
        "X_te = vectorizer.transform(test_clean['text'].astype(str))\n\n"
        "model = XGBRegressor(\n"
        "    objective='reg:squarederror',\n"
        "    n_estimators=250,\n"
        "    learning_rate=0.05,\n"
        "    max_depth=8,\n"
        "    subsample=0.8,\n"
        "    colsample_bytree=0.8,\n"
        "    random_state=42,\n"
        "    n_jobs=-1,\n"
        "    eval_metric='mae',\n"
        ")\n\n"
        "model.fit(X_tr, y_train)\n\n"
        "val_pred = model.predict(X_va)\n"
        "test_pred = model.predict(X_te)\n\n"
        "print(f'Validation MAE: {mean_absolute_error(y_val, val_pred):.4f}')\n"
        "print(f'Test MAE: {mean_absolute_error(test_clean['rating'].astype(float), test_pred):.4f}')\n"
        "print(f'Test RMSE: {np.sqrt(mean_squared_error(test_clean['rating'].astype(float), test_pred)):.4f}')\n"
        "print(f'Test R^2: {r2_score(test_clean['rating'].astype(float), test_pred):.4f}')"
    ),
    nbf.v4.new_markdown_cell('\n---\n\n## Step 3: Save metrics\n\nWe export validation and test metrics as a CSV in the artifacts folder.'),
    nbf.v4.new_code_cell(
        "metrics_df = pd.DataFrame({\n"
        "    'set': ['validation', 'test'],\n"
        "    'mae': [\n"
        "        mean_absolute_error(y_val, val_pred),\n"
        "        mean_absolute_error(test_clean['rating'].astype(float), test_pred),\n"
        "    ],\n"
        "    'rmse': [\n"
        "        np.sqrt(mean_squared_error(y_val, val_pred)),\n"
        "        np.sqrt(mean_squared_error(test_clean['rating'].astype(float), test_pred)),\n"
        "    ],\n"
        "    'r2': [\n"
        "        r2_score(y_val, val_pred),\n"
        "        r2_score(test_clean['rating'].astype(float), test_pred),\n"
        "    ],\n"
        "})\n\n"
        "metrics_path = os.path.join(ARTIFACTS_DIR, 'metrics', 'xgboost_regression_metrics.csv')\n"
        "os.makedirs(os.path.dirname(metrics_path), exist_ok=True)\n"
        "metrics_df.to_csv(metrics_path, index=False)\n\n"
        "print(metrics_df.to_string(index=False))"
    ),
]

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f'Notebook written: {notebook_path}')
print(f'Train cleaned CSV: {artifacts_dir / "xgboost_regression_train_clean.csv"}')
print(f'Test cleaned CSV: {artifacts_dir / "xgboost_regression_test_clean.csv"}')
print(f'Metrics CSV: {metrics_dir / "xgboost_regression_metrics.csv"}')
print(f'Validation MAE: {mean_absolute_error(y_val, val_pred):.4f}')
print(f'Test MAE: {mean_absolute_error(test_df["rating"].astype(float), test_pred):.4f}')
print(f'Test RMSE: {np.sqrt(mean_squared_error(test_df["rating"].astype(float), test_pred)):.4f}')
print(f'Test R2: {r2_score(test_df["rating"].astype(float), test_pred):.4f}')
