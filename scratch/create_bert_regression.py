import nbformat as nbf
import os

def create_bert_regression_notebook():
    nb = nbf.v4.new_notebook()

    cells = []

    # Markdown
    cells.append(nbf.v4.new_markdown_cell("""# Data Speaks!! — BERT Fine-Tuning for Rating Regression

This notebook fine-tunes **`bert-base-uncased`** on the IMDB movie review dataset for **Continuous Rating Regression** (predicting a score from 1 to 10).

### Pipeline Steps
| Step | Description |
|------|-------------|
| 0 | Environment & directory setup |
| 1 | Load & verify raw data (25K train + 25K test) |
| 2 | Data audit — nulls, duplicates, text stats, HTML artifacts |
| 3 | Text cleaning — HTML strip, whitespace normalize, dedup |
| 4 | Stratified train/validation split (90/10, seed=42) |
| 5 | BERT tokenization & PyTorch DataLoaders |
| 6 | BERT model training (3 epochs, GPU) & test-set evaluation (MSE / MAE) |
| 7 | Export metrics CSV |

All artifacts are saved under **`artifacts/`**.
"""))

    # Step 0
    cells.append(nbf.v4.new_markdown_cell("""---
## Step 0 — Environment & Directory Setup"""))
    cells.append(nbf.v4.new_code_cell("""import os
import sys
import time
import re
import html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

# ── Reproducibility ────────
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

# ── Paths ─────────────
DATA_DIR    = os.path.join("..", "..", "processed_for_regression")
ARTIFACTS   = os.path.join(".", "artifacts")
MODELS_DIR  = os.path.join(ARTIFACTS, "models")
METRICS_DIR = os.path.join(ARTIFACTS, "metrics")
PLOTS_DIR   = os.path.join(ARTIFACTS, "plots")

for d in [ARTIFACTS, MODELS_DIR, METRICS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Device ─────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device  : {device}")
if torch.cuda.is_available():
    print(f"GPU     : {torch.cuda.get_device_name(0)}")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM    : {vram_gb:.1f} GB")
else:
    print("⚠  No GPU detected — training will be very slow on CPU.")
print(f"PyTorch : {torch.__version__}")
"""))

    # Step 1
    cells.append(nbf.v4.new_markdown_cell("""---
## Step 1 — Acquire and Verify Data"""))
    cells.append(nbf.v4.new_code_cell("""train_df = pd.read_csv(os.path.join(DATA_DIR, "imdb_regression_train.csv"))
test_df  = pd.read_csv(os.path.join(DATA_DIR, "imdb_regression_test.csv"))

print(f"Train shape   : {train_df.shape}")
print(f"Test  shape   : {test_df.shape}")
print(f"Columns       : {list(train_df.columns)}")

print(f"\\nTrain rating counts:\\n{train_df['rating'].value_counts().sort_index()}")
print(f"\\nTest  rating counts:\\n{test_df['rating'].value_counts().sort_index()}")
"""))

    # Step 2
    cells.append(nbf.v4.new_markdown_cell("""---
## Step 2 — Data Audit (Read-Only)"""))
    cells.append(nbf.v4.new_code_cell("""#Null 
print("Counting null values")
print(f"Train nulls:\\n{train_df.isnull().sum()}")
print(f"\\nTest nulls:\\n{test_df.isnull().sum()}")

#Duplicate
train_dups = train_df.duplicated(subset=["text"]).sum()
test_dups  = test_df.duplicated(subset=["text"]).sum()
print(f"\\nCounting Duplicates")
print(f"Train: {train_dups}  |  Test: {test_dups}")

# Text length
train_len = train_df["text"].str.len()
test_len  = test_df["text"].str.len()
print(f"\\nStatistics of character length")
print(f"Train : min={train_len.min()}, max={train_len.max()}, "
      f"mean={train_len.mean():.1f}, median={train_len.median():.1f}")
print(f"Test  : min={test_len.min()}, max={test_len.max()}, "
      f"mean={test_len.mean():.1f}, median={test_len.median():.1f}")

# HTML
html_train = train_df["text"].str.contains("<br", regex=False).sum()
html_test  = test_df["text"].str.contains("<br", regex=False).sum()
print(f"\\nChecking html tags")
print(f"Train: {html_train} ({html_train/len(train_df):.1%})")
print(f"Test:  {html_test}  ({html_test/len(test_df):.1%})")
"""))

    # Step 3
    cells.append(nbf.v4.new_markdown_cell("""---
## Step 3 — Clean the Text"""))
    cells.append(nbf.v4.new_code_cell("""def clean_text(text: str) -> str:
    \"\"\"Strip HTML tags/entities, normalize whitespace.\"\"\"
    text = html.unescape(text)                                    # decode &amp; etc.
    text = re.sub(r"<br\\s*/?>", " ", text, flags=re.IGNORECASE)  # <br /> → space
    text = re.sub(r"<[^>]+>", " ", text)                         # any remaining tags
    text = re.sub(r"\\s+", " ", text).strip()                     # collapse whitespace
    return text

train_df["text_clean"] = train_df["text"].apply(clean_text)
test_df["text_clean"]  = test_df["text"].apply(clean_text)

# Remove exact duplicates (keep first occurrence)
train_clean = (
    train_df.drop_duplicates(subset=["text_clean"], keep="first")
    .copy()[["text_clean", "rating"]]
    .rename(columns={"text_clean": "text"})
)
test_clean = (
    test_df.drop_duplicates(subset=["text_clean"], keep="first")
    .copy()[["text_clean", "rating"]]
    .rename(columns={"text_clean": "text"})
)

print(f"Train rows: {len(train_df)} → {len(train_clean)} (removed {len(train_df)-len(train_clean)} duplicates)")
print(f"Test  rows: {len(test_df)}  → {len(test_clean)}  (removed {len(test_df)-len(test_clean)} duplicates)")

# Save cleaned datasets
train_clean.to_csv(os.path.join(ARTIFACTS, "imdb_regression_train_clean.csv"), index=False)
test_clean.to_csv(os.path.join(ARTIFACTS, "imdb_regression_test_clean.csv"), index=False)
print(f"Saved cleaned CSVs to {ARTIFACTS}/")
"""))

    # Step 4
    cells.append(nbf.v4.new_markdown_cell("""---
## Step 4 — Train / Validation Split"""))
    cells.append(nbf.v4.new_code_cell("""# For regression, we can still stratify by the discrete rating values since they are integers from 1-10.
train_split, val_split = train_test_split(
    train_clean,
    test_size=0.10,
    random_state=SEED,
    stratify=train_clean["rating"],
)

print(f"Train subset : {len(train_split):>6}  ({len(train_split)/len(train_clean):.1%})")
print(f"Val   subset : {len(val_split):>6}  ({len(val_split)/len(train_clean):.1%})")
print(f"Test  (held) : {len(test_clean):>6}")
"""))

    # Step 5
    cells.append(nbf.v4.new_markdown_cell("""---
## Step 5 — BERT Tokenization & DataLoaders"""))
    cells.append(nbf.v4.new_code_cell("""MODEL_NAME = "bert-base-uncased"
tokenizer  = BertTokenizerFast.from_pretrained(MODEL_NAME)

class IMDBRegressionDataset(Dataset):
    \"\"\"PyTorch Dataset that tokenizes on-the-fly. For Regression, labels must be float.\"\"\"

    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts     = list(texts)
        self.labels    = list(labels)
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        # Important: Use float for regression
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

BATCH_SIZE = 16

train_ds = IMDBRegressionDataset(train_split["text"].values, train_split["rating"].values, tokenizer)
val_ds   = IMDBRegressionDataset(val_split["text"].values,   val_split["rating"].values,   tokenizer)
test_ds  = IMDBRegressionDataset(test_clean["text"].values,  test_clean["rating"].values,  tokenizer)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

print(f"Tokenizer vocab : {tokenizer.vocab_size}")
print(f"Dataset sizes   : train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
print(f"Batch size      : {BATCH_SIZE}")
print(f"Batches         : train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")
"""))

    # Step 6
    cells.append(nbf.v4.new_markdown_cell("""---
## Step 6 — Train BERT (3 Epochs, Mixed Precision)"""))
    cells.append(nbf.v4.new_code_cell("""# Note: num_labels=1 tells HuggingFace to use MSELoss for regression
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
model.to(device)

EPOCHS      = 3
LR          = 2e-5
total_steps = len(train_loader) * EPOCHS

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * 0.1),
    num_training_steps=total_steps,
)
scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

history = {"train_loss": [], "val_loss": [], "val_mae": []}

print(f"Training {MODEL_NAME} for {EPOCHS} epochs on {device}…")
print(f"Total optimiser steps: {total_steps}")
t0 = time.time()

for epoch in range(EPOCHS):
    t_epoch = time.time()

    # Training
    model.train()
    running_loss = 0.0

    for step, batch in enumerate(train_loader, 1):
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbls = batch["labels"].to(device)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            out  = model(ids, attention_mask=mask, labels=lbls)
            loss = out.loss # This is MSELoss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        running_loss += loss.item()

        if step % 300 == 0 or step == len(train_loader):
            print(f"  Epoch {epoch+1}/{EPOCHS}  step {step}/{len(train_loader)}  "
                  f"batch_loss(MSE)={loss.item():.4f}")

    avg_train = running_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss_sum = 0.0
    vp, vt = [], []

    with torch.no_grad():
        for batch in val_loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbls = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                out = model(ids, attention_mask=mask, labels=lbls)

            val_loss_sum += out.loss.item()
            # For regression, logits is shape (batch_size, 1)
            vp.extend(out.logits.squeeze(1).cpu().numpy())
            vt.extend(lbls.cpu().numpy())

    avg_val = val_loss_sum / len(val_loader)
    v_mae   = mean_absolute_error(vt, vp)

    history["train_loss"].append(avg_train)
    history["val_loss"].append(avg_val)
    history["val_mae"].append(v_mae)

    dt = time.time() - t_epoch
    print(f"  ── Epoch {epoch+1} done in {dt:.0f}s │ "
          f"train_loss(MSE)={avg_train:.4f}  val_loss(MSE)={avg_val:.4f}  val_MAE={v_mae:.4f}\\n")

total_wall_time = time.time() - t0
print(f"Total training wall-clock: {total_wall_time:.1f}s  ({total_wall_time/60:.1f} min)")
"""))

    # Step 7
    cells.append(nbf.v4.new_markdown_cell("""---
## Step 7 — Final Evaluation & Export Metrics"""))
    cells.append(nbf.v4.new_code_cell("""# Final Evaluation on the held-out TEST set
model.eval()
test_preds, test_truths = [], []

print("Running evaluation on test set...")
with torch.no_grad():
    for batch in test_loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbls = batch["labels"].to(device)

        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            out = model(ids, attention_mask=mask)
            
        test_preds.extend(out.logits.squeeze(1).cpu().numpy())
        test_truths.extend(lbls.cpu().numpy())

test_mse = mean_squared_error(test_truths, test_preds)
test_mae = mean_absolute_error(test_truths, test_preds)
test_r2  = r2_score(test_truths, test_preds)

print(f"\\nTest MSE : {test_mse:.4f}")
print(f"Test MAE : {test_mae:.4f}")
print(f"Test R^2 : {test_r2:.4f}")

# Save metrics
metrics_df = pd.DataFrame([{
    "model": "BERT_Regression",
    "test_mse": test_mse,
    "test_mae": test_mae,
    "test_r2": test_r2
}])
metrics_df.to_csv(os.path.join(METRICS_DIR, "bert_regression_metrics.csv"), index=False)
print(f"\\nSaved metrics to {METRICS_DIR}/bert_regression_metrics.csv")
"""))

    nb.cells = cells
    
    out_dir = r"c:\Users\dhanu\.gemini\antigravity-ide\scratch\msd_project\regression_models\bert"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "BERT_REGRESSION.ipynb")
    
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
        
    print(f"Created notebook at {out_path}")

if __name__ == "__main__":
    create_bert_regression_notebook()
