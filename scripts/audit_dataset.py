import hashlib
from pathlib import Path
from PIL import Image
import pandas as pd

meta_path = Path("data/species_images/metadata.csv")
df = pd.read_csv(meta_path)
print(f"Total metadata rows: {len(df)}")
print("Columns:", df.columns.tolist())
print("\nClass counts in metadata:")
print(df["species"].value_counts())

print("\nLicenses:")
print(df["license"].value_counts())

# Check gbif_key duplicates
dup_keys = df[df.duplicated(subset=["gbif_key"], keep=False)]
print(f"\nDuplicate gbif_key occurrences: {len(dup_keys)}")
if len(dup_keys) > 0:
    print(dup_keys[["species", "filename", "gbif_key"]])

# Check observation grouping
unique_obs = df["gbif_key"].nunique()
print(f"Unique GBIF occurrence keys: {unique_obs} out of {len(df)} images")

# Verify physical files
file_status = []
md5_hashes = {}
duplicate_hashes = []

for idx, row in df.iterrows():
    p = Path(row["filepath"])
    exists = p.exists()
    size = p.stat().st_size if exists else 0
    is_valid_img = False
    dims = (0, 0)
    img_mode = ""
    h = ""
    if exists:
        try:
            with open(p, "rb") as f:
                content = f.read()
                h = hashlib.md5(content).hexdigest()
            if h in md5_hashes:
                duplicate_hashes.append((row["filename"], md5_hashes[h], row["species"]))
            else:
                md5_hashes[h] = row["filename"]

            with Image.open(p) as img:
                dims = img.size
                img_mode = img.mode
                is_valid_img = True
        except Exception as e:
            is_valid_img = False
    
    file_status.append({
        "filename": row["filename"],
        "species": row["species"],
        "exists": exists,
        "size_bytes": size,
        "valid": is_valid_img,
        "dims": dims,
        "mode": img_mode
    })

fdf = pd.DataFrame(file_status)
print(f"\nTotal verified files on disk: {len(fdf)}")
print(f"Total valid PIL images: {fdf['valid'].sum()}")
print(f"Duplicate MD5 images: {len(duplicate_hashes)}")
if duplicate_hashes:
    for d in duplicate_hashes:
        print("  Duplicate hash:", d)

print("\nMin/Max dimensions:")
print("Min width:", fdf["dims"].apply(lambda d: d[0]).min(), "Max width:", fdf["dims"].apply(lambda d: d[0]).max())
print("Min height:", fdf["dims"].apply(lambda d: d[1]).min(), "Max height:", fdf["dims"].apply(lambda d: d[1]).max())
print("Min size bytes:", fdf["size_bytes"].min(), "Max size bytes:", fdf["size_bytes"].max())
