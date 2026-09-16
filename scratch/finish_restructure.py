import os
import glob
import json
import shutil
import csv

BASE_DIR = r"c:\Users\dhanu\.gemini\antigravity-ide\scratch\msd_project"

def finish_restructure():
    binary_models_dir = os.path.join(BASE_DIR, "binary_classification_models")
    
    # Move bert
    src_bert = os.path.join(BASE_DIR, "bert")
    dst_bert = os.path.join(binary_models_dir, "bert")
    if os.path.exists(src_bert) and not os.path.exists(dst_bert):
        try:
            shutil.move(src_bert, dst_bert)
        except Exception as e:
            print(f"Warning moving bert: {e}")
            
    # Try removing old lstm if it was successfully copied
    src_lstm = os.path.join(BASE_DIR, "lstm")
    if os.path.exists(src_lstm):
        try:
            shutil.rmtree(src_lstm)
        except Exception:
            pass

def update_notebook_paths():
    binary_models_dir = os.path.join(BASE_DIR, "binary_classification_models")
    notebooks = glob.glob(os.path.join(binary_models_dir, "**", "*.ipynb"), recursive=True)
    
    for nb_path in notebooks:
        try:
            with open(nb_path, "r", encoding="utf-8") as f:
                nb = json.load(f)
                
            changed = False
            for cell in nb.get("cells", []):
                if cell["cell_type"] == "code":
                    new_source = []
                    for line in cell["source"]:
                        if '"../processed/imdb_train.csv"' in line or "'../processed/imdb_train.csv'" in line:
                            line = line.replace("../processed/imdb_train.csv", "../../processed_for_binary_classification/imdb_train.csv")
                            changed = True
                        if '"../processed/imdb_test.csv"' in line or "'../processed/imdb_test.csv'" in line:
                            line = line.replace("../processed/imdb_test.csv", "../../processed_for_binary_classification/imdb_test.csv")
                            changed = True
                        new_source.append(line)
                    cell["source"] = new_source
            
            if changed:
                with open(nb_path, "w", encoding="utf-8") as f:
                    json.dump(nb, f, indent=1)
                print(f"Updated paths in {os.path.basename(nb_path)}")
        except Exception as e:
            print(f"Error updating {nb_path}: {e}")

def create_regression_dataset():
    acl_dir = os.path.join(BASE_DIR, "aclImdb")
    out_dir = os.path.join(BASE_DIR, "processed_for_regression")
    os.makedirs(out_dir, exist_ok=True)
    
    for split in ["train", "test"]:
        out_csv = os.path.join(out_dir, f"imdb_regression_{split}.csv")
        print(f"Generating {out_csv}...")
        
        with open(out_csv, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["text", "rating"])
            
            for sentiment in ["pos", "neg"]:
                folder_path = os.path.join(acl_dir, split, sentiment)
                if not os.path.exists(folder_path):
                    continue
                
                for filename in os.listdir(folder_path):
                    if filename.endswith(".txt"):
                        parts = filename.replace(".txt", "").split("_")
                        if len(parts) == 2:
                            rating = int(parts[1])
                            filepath = os.path.join(folder_path, filename)
                            with open(filepath, "r", encoding="utf-8") as f:
                                text = f.read().strip()
                            writer.writerow([text, rating])
        print(f"Successfully created {split} regression dataset.")

if __name__ == "__main__":
    finish_restructure()
    update_notebook_paths()
    create_regression_dataset()
    print("Done!")
