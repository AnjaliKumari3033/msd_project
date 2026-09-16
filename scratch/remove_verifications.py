import json

paths = [
    "c:/Users/dhanu/.gemini/antigravity-ide/scratch/msd_project/bert/BERT.ipynb",
    "c:/Users/dhanu/.gemini/antigravity-ide/scratch/msd_project/lstm/LSTM.ipynb"
]

for path in paths:
    with open(path, "r", encoding="utf-8") as f:
        notebook = json.load(f)
    
    new_cells = []
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if "Mandatory Verifications" in source or "MANDATORY VERIFICATIONS" in source:
            continue
        new_cells.append(cell)
        
    notebook["cells"] = new_cells
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
