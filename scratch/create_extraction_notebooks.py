import nbformat as nbf
import os

def create_notebooks():
    # ---------------------------------------------------------
    # 2. Regression Notebook
    # ---------------------------------------------------------
    nb_reg = nbf.v4.new_notebook()
    
    cells_reg = []
    cells_reg.append(nbf.v4.new_markdown_cell("""# Extracting the IMDB Regression Dataset

This notebook demonstrates how the **Continuous Rating Dataset** was extracted from the raw `aclImdb` folder. 
Unlike the binary classification task (where we just use Positive/Negative), the raw IMDB dataset actually encodes the exact rating (from 1 to 10) in the filename!

The format of each text file in the raw dataset is: `[id]_[rating].txt`

We will extract the `rating` from the filename and pair it with the review text.
"""))
    
    cells_reg.append(nbf.v4.new_code_cell("""import os
import glob
import pandas as pd
from tqdm.notebook import tqdm

# Define paths to the raw data
# Note: Ensure you have downloaded and extracted the aclImdb dataset into a folder named 'aclImdb'
base_path = "../aclImdb"

def extract_regression_data(split):
    \"\"\"
    Iterates through both 'pos' and 'neg' folders for a given split (train/test),
    extracts the text and the 1-10 rating from the filename.
    \"\"\"
    data = []
    
    for sentiment in ['pos', 'neg']:
        folder_path = os.path.join(base_path, split, sentiment)
        if not os.path.exists(folder_path):
            print(f"Directory not found: {folder_path}")
            continue
            
        file_paths = glob.glob(os.path.join(folder_path, "*.txt"))
        
        for file_path in tqdm(file_paths, desc=f"Processing {split}/{sentiment}"):
            # Filename format: id_rating.txt (e.g., 1234_7.txt)
            filename = os.path.basename(file_path)
            rating_str = filename.split('_')[1].replace('.txt', '')
            rating = int(rating_str)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
            data.append({"text": text, "rating": rating})
            
    return pd.DataFrame(data)
"""))

    cells_reg.append(nbf.v4.new_code_cell("""# Extract Train and Test data
print("Extracting Training Data...")
train_df = extract_regression_data("train")

print("Extracting Testing Data...")
test_df = extract_regression_data("test")

print(f"\\nTrain dataset shape: {train_df.shape}")
print(f"Test dataset shape: {test_df.shape}")
"""))

    cells_reg.append(nbf.v4.new_code_cell("""# Preview the extracted regression dataset
train_df.head()
"""))

    cells_reg.append(nbf.v4.new_code_cell("""# Ensure the output directory exists
output_dir = "../processed_for_regression"
os.makedirs(output_dir, exist_ok=True)

# Save to CSV
train_csv_path = os.path.join(output_dir, "imdb_regression_train.csv")
test_csv_path = os.path.join(output_dir, "imdb_regression_test.csv")

train_df.to_csv(train_csv_path, index=False)
test_df.to_csv(test_csv_path, index=False)

print(f"Saved regression data to {output_dir}")
"""))
    
    nb_reg.cells = cells_reg
    
    # ---------------------------------------------------------
    # 3. Unsupervised Notebook
    # ---------------------------------------------------------
    nb_unsup = nbf.v4.new_notebook()
    
    cells_unsup = []
    cells_unsup.append(nbf.v4.new_markdown_cell("""# Extracting the IMDB Unsupervised Dataset

This notebook demonstrates how the **Unlabelled Dataset** was extracted from the raw `aclImdb` folder. 
The IMDB dataset provides 50,000 extra unlabelled reviews specifically for unsupervised learning tasks like clustering or topic modeling.

These files are located in `aclImdb/train/unsup`. They do not have positive/negative labels, nor do they have 1-10 ratings.
"""))
    
    cells_unsup.append(nbf.v4.new_code_cell("""import os
import glob
import pandas as pd
from tqdm.notebook import tqdm

# Define path to the raw unsupervised data
unsup_folder = "../aclImdb/train/unsup"

def extract_unsup_data():
    data = []
    
    if not os.path.exists(unsup_folder):
        print(f"Directory not found: {unsup_folder}")
        return pd.DataFrame(data)
        
    file_paths = glob.glob(os.path.join(unsup_folder, "*.txt"))
    
    for file_path in tqdm(file_paths, desc="Processing unsupervised data"):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # Since these are unlabelled, we just save the text
        data.append({"text": text})
        
    return pd.DataFrame(data)
"""))

    cells_unsup.append(nbf.v4.new_code_cell("""# Extract the data
print("Extracting Unsupervised Data...")
unsup_df = extract_unsup_data()

print(f"\\nUnsupervised dataset shape: {unsup_df.shape}")
"""))

    cells_unsup.append(nbf.v4.new_code_cell("""# Preview the extracted unsupervised dataset
unsup_df.head()
"""))

    cells_unsup.append(nbf.v4.new_code_cell("""# Ensure the output directory exists
output_dir = "../processed_for_clustering"
os.makedirs(output_dir, exist_ok=True)

# Save to CSV
unsup_csv_path = os.path.join(output_dir, "imdb_unsup.csv")

unsup_df.to_csv(unsup_csv_path, index=False)

print(f"Saved unsupervised data to {output_dir}")
"""))

    nb_unsup.cells = cells_unsup
    
    # Save notebooks
    os.makedirs(r"c:\Users\dhanu\.gemini\antigravity-ide\scratch\msd_project\notebooks", exist_ok=True)
    
    with open(r"c:\Users\dhanu\.gemini\antigravity-ide\scratch\msd_project\notebooks\2_extract_regression_data.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb_reg, f)
        
    with open(r"c:\Users\dhanu\.gemini\antigravity-ide\scratch\msd_project\notebooks\3_extract_unsupervised_data.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb_unsup, f)

if __name__ == "__main__":
    create_notebooks()
