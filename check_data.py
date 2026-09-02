import pandas as pd
import os

# Load the training CSV
df = pd.read_csv("train.csv")

print("Shape of train.csv:", df.shape)
print("\nColumn names:", list(df.columns))
print("\nFirst 5 rows:")
print(df.head())

print("\nClass distribution (diagnosis column, if present):")
if "diagnosis" in df.columns:
    print(df["diagnosis"].value_counts().sort_index())

# Check that image files actually exist
image_folder = "train_images"
files_in_folder = set(os.listdir(image_folder))

# Guess the id column (APTOS usually calls it 'id_code')
id_col = "id_code" if "id_code" in df.columns else df.columns[0]

missing = []
for img_id in df[id_col]:
    # try common extensions
    found = any(f"{img_id}{ext}" in files_in_folder for ext in [".png", ".jpg", ".jpeg"])
    if not found:
        missing.append(img_id)

print(f"\nTotal rows in CSV: {len(df)}")
print(f"Total images in folder: {len(files_in_folder)}")
print(f"Missing images: {len(missing)}")
if missing[:5]:
    print("Example missing IDs:", missing[:5])