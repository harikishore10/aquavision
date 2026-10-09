"""Download real fish species images from GBIF Open Biodiversity data.

Retrieves verified research observations with photos for:
- tilapia: Oreochromis niloticus
- catfish: Clarias gariepinus
- carp: Cyprinus carpio
- milkfish: Chanos chanos
- pangasius: Pangasianodon hypophthalmus
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request
from PIL import Image

TARGET_SPECIES = {
    "tilapia": "Oreochromis niloticus",
    "catfish": "Clarias gariepinus",
    "carp": "Cyprinus carpio",
    "milkfish": "Chanos chanos",
    "pangasius": "Pangasianodon hypophthalmus",
}

IMAGES_PER_SPECIES = 35
OUTPUT_DIR = Path("data/species_images")


def fetch_gbif_records(scientific_name: str, limit: int = 60) -> list[dict]:
    query_url = (
        f"https://api.gbif.org/v1/occurrence/search?"
        f"scientificName={urllib.parse.quote(scientific_name)}&"
        f"mediaType=StillImage&limit={limit}"
    )
    req = urllib.request.Request(query_url, headers={"User-Agent": "AquaVision-DatasetCollector/1.0"})
    with urllib.request.urlopen(req, timeout=12) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data.get("results", [])


def download_image(url: str, dest_path: Path, timeout: float = 10.0) -> bool:
    try:
        # Prefer medium resolution on iNaturalist CDN if available
        cdn_url = url
        if "original.jpg" in url:
            cdn_url = url.replace("original.jpg", "medium.jpg")
        elif "original.jpeg" in url:
            cdn_url = url.replace("original.jpeg", "medium.jpeg")

        req = urllib.request.Request(cdn_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content = resp.read()

        # Validate with PIL
        img = Image.open(io.BytesIO(content))
        img.verify()

        # Reopen and save as clean RGB JPEG
        img = Image.open(io.BytesIO(content)).convert("RGB")
        if img.width < 64 or img.height < 64:
            return False

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(dest_path, "JPEG", quality=90)
        return True
    except Exception:
        return False


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata_rows = []

    print("Starting GBIF research dataset acquisition for AquaVision...")

    for label, sciname in TARGET_SPECIES.items():
        species_dir = OUTPUT_DIR / label
        species_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nQuerying GBIF for {label} ({sciname})...")
        records = fetch_gbif_records(sciname, limit=70)
        print(f"  Found {len(records)} observation candidates with media.")

        downloaded = 0
        for rec in records:
            if downloaded >= IMAGES_PER_SPECIES:
                break

            media_list = rec.get("media", [])
            if not media_list:
                continue

            img_url = media_list[0].get("identifier")
            if not img_url:
                continue

            gbif_id = rec.get("key", f"{label}_{downloaded}")
            license_val = rec.get("license", "CC-BY-4.0")
            filename = f"{label}_{downloaded + 1:03d}.jpg"
            dest_file = species_dir / filename

            success = download_image(img_url, dest_file)
            if success:
                downloaded += 1
                metadata_rows.append({
                    "species": label,
                    "scientific_name": sciname,
                    "filename": filename,
                    "filepath": str(dest_file.as_posix()),
                    "gbif_key": gbif_id,
                    "license": license_val,
                    "source_url": img_url,
                })
                print(f"  [{label}] Downloaded {downloaded}/{IMAGES_PER_SPECIES}: {filename}")
                time.sleep(0.1)

        print(f"Completed {label}: {downloaded} valid images saved.")

    # Save metadata CSV
    meta_path = OUTPUT_DIR / "metadata.csv"
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["species", "scientific_name", "filename", "filepath", "gbif_key", "license", "source_url"],
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"\nDataset acquisition finished! Total images: {len(metadata_rows)}")
    print(f"Metadata manifest saved to {meta_path}")


if __name__ == "__main__":
    main()
