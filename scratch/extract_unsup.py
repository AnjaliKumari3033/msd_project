import os
import pandas as pd

def extract_unsup_data():
    base_dir = r"c:\Users\dhanu\.gemini\antigravity-ide\scratch\msd_project"
    unsup_dir = os.path.join(base_dir, "aclImdb", "train", "unsup")
    out_dir = os.path.join(base_dir, "processed_for_clustering")
    
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Reading files from {unsup_dir}...")
    reviews = []
    
    # Check if directory exists
    if not os.path.exists(unsup_dir):
        print(f"Error: {unsup_dir} does not exist.")
        return
        
    for filename in os.listdir(unsup_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(unsup_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                
            # For unsupervised data, we don't have a label. 
            # We can just store the text and the original filename/id if we want.
            # Filename format is usually [id]_0.txt for unsup.
            review_id = filename.split('_')[0]
            
            reviews.append({
                "id": review_id,
                "text": text
            })
            
    print(f"Loaded {len(reviews)} unsupervised reviews.")
    
    df = pd.DataFrame(reviews)
    out_file = os.path.join(out_dir, "imdb_unsup.csv")
    df.to_csv(out_file, index=False, encoding="utf-8")
    
    print(f"Saved successfully to {out_file}")

if __name__ == "__main__":
    extract_unsup_data()
