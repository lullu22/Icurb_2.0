import os
import json
import random

# ===============================
# CONFIGURATION
# ===============================
folders = [
   "/home/c-lcuffaro/Desktop/dataset_prova/cropped_tiff"
    
]

# Percentuali di split
split_percentages = {
    "pretrain": 0.00,
    "train": 0.00,
    "valid": 0.00,
    "test": 1.00
}

output_json = "/home/c-lcuffaro/Desktop/data_split.json"

# ===============================
# COLLECT FILES
# ===============================
all_files = []
for folder in folders:
    for root, _, files in os.walk(folder):
        tiff_files = [f for f in files if f.lower().endswith(".tiff")]
        all_files.extend(tiff_files)

if len(all_files) == 0:
    raise ValueError("Nessun file .tiff trovato nelle cartelle specificate.")

# Estrai solo il nome senza estensione
all_files = [os.path.splitext(f)[0] for f in all_files]

# ===============================
# SHUFFLE FILES
# ===============================
random.shuffle(all_files)

# ===============================
# SPLIT FILES
# ===============================
n_total = len(all_files)
n_pretrain = int(n_total * split_percentages["pretrain"])
n_train = int(n_total * split_percentages["train"])
n_valid = int(n_total * split_percentages["valid"])
n_test = n_total - (n_pretrain + n_train + n_valid)

dataset_split = {
    "pretrain": all_files[:n_pretrain],
    "train": all_files[n_pretrain:n_pretrain+n_train],
    "valid": all_files[n_pretrain+n_train:n_pretrain+n_train+n_valid],
    "test": all_files[n_pretrain+n_train+n_valid:]
}

# ===============================
# SAVE TO JSON
# ===============================
with open(output_json, "w") as f:
    json.dump(dataset_split, f, indent=4)

print(f"Dataset split salvato in {output_json}")
print(f"Totale file: {n_total}")
for k, v in dataset_split.items():
    print(f"{k}: {len(v)}")


