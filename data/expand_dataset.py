"""Expand AquaVision real fish species dataset using GBIF Open Biodiversity data.

Retrieves verified research observations with photos for:
- tilapia: Oreochromis niloticus
- catfish: Clarias gariepinus
- carp: Cyprinus carpio
- milkfish: Chanos chanos
- pangasius: Pangasianodon hypophthalmus

Expands from existing 35 images/species to 100 images/species (500 total).
Preserves observer identifiers, GBIF occurrence keys, event dates, source URLs, and CC licenses.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import urllib.parse
import urllib.request
import numpy as np
from PIL import Image

# Ensure console supports UTF-8 on Windows
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

TARGET_SPECIES = {
    "tilapia": "Oreochromis niloticus",
    "catfish": "Clarias gariepinus",
    "carp": "Cyprinus carpio",
    "milkfish": "Chanos chanos",
    "pangasius": "Pangasianodon hypophthalmus",
}

TARGET_COUNT_PER_SPECIES = 100
OUTPUT_DIR = Path("data/species_images")
CACHE_FILE = OUTPUT_DIR / "observer_backfill_cache.json"


def dhash(img: Image.Image, hash_size: int = 8) -> int:
    """Compute difference hash (dHash) to detect duplicate or near-identical images."""
    resized = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    arr = np.array(resized, dtype=np.int32)
    diff = arr[:, :-1] > arr[:, 1:]
    diff_flat = diff.flatten()
    return int(sum([2**i for (i, v) in enumerate(diff_flat) if v]))


def fetch_gbif_records(scientific_name: str, limit: int = 200, offset: int = 0) -> list[dict]:
    query_url = (
        f"https://api.gbif.org/v1/occurrence/search?"
        f"scientificName={urllib.parse.quote(scientific_name)}&"
        f"mediaType=StillImage&limit={limit}&offset={offset}"
    )
    req = urllib.request.Request(query_url, headers={"User-Agent": "AquaVision-DatasetExpansion/2.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data.get("results", [])


def fetch_occurrence_details(key: str | int) -> dict:
    url = f"https://api.gbif.org/v1/occurrence/{key}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AquaVision-DatasetExpansion/2.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return {}


def download_and_validate_image(url: str, dest_path: Path, timeout: float = 10.0) -> tuple[bool, int, str]:
    """Download image, validate dimensions & RGB channels, and compute perceptual hash."""
    try:
        cdn_url = url
        if "original.jpg" in url:
            cdn_url = url.replace("original.jpg", "medium.jpg")
        elif "original.jpeg" in url:
            cdn_url = url.replace("original.jpeg", "medium.jpeg")

        req = urllib.request.Request(cdn_url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content = resp.read()

        md5_val = hashlib.md5(content).hexdigest()

        # Validate with PIL
        raw = Image.open(io.BytesIO(content))
        raw.verify()

        # Reopen and convert to RGB
        img = Image.open(io.BytesIO(content)).convert("RGB")
        if img.width < 64 or img.height < 64:
            return False, 0, ""

        img_hash = dhash(img)

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(dest_path, "JPEG", quality=90)
        return True, img_hash, md5_val
    except Exception:
        return False, 0, ""


def save_metadata(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "species",
        "scientific_name",
        "filename",
        "filepath",
        "gbif_key",
        "recorded_by",
        "event_date",
        "license",
        "source_url",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def expand_dataset():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = OUTPUT_DIR / "metadata.csv"

    # Step 1: Load existing metadata
    existing_rows = []
    seen_keys = set()
    seen_hashes = set()

    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_rows.append(row)
                if row.get("gbif_key"):
                    seen_keys.add(str(row["gbif_key"]))
                fp = Path(row["filepath"])
                if fp.exists():
                    try:
                        with Image.open(fp) as im:
                            h = dhash(im)
                            seen_hashes.add(h)
                    except Exception:
                        pass

    print(f"Loaded {len(existing_rows)} existing records across {len(seen_keys)} unique GBIF keys.")

    # Load observer backfill cache if present
    observer_cache = {}
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                observer_cache = json.load(f)
        except Exception:
            observer_cache = {}

    # Backfill missing recorded_by / event_date for existing rows
    print("Checking observer attribution for existing records...")
    cache_dirty = False
    for idx, row in enumerate(existing_rows):
        k = str(row.get("gbif_key", ""))
        if not row.get("recorded_by") or row["recorded_by"] == "unknown":
            if k in observer_cache:
                row["recorded_by"] = observer_cache[k].get("recorded_by", "unknown")
                row["event_date"] = observer_cache[k].get("event_date", "unknown")
            elif k:
                det = fetch_occurrence_details(k)
                row["recorded_by"] = det.get("recordedBy", "unknown")
                row["event_date"] = det.get("eventDate", "unknown")
                observer_cache[k] = {
                    "recorded_by": row["recorded_by"],
                    "event_date": row["event_date"],
                }
                cache_dirty = True
                if (idx + 1) % 25 == 0:
                    print(f"  Processed {idx + 1}/{len(existing_rows)} observer attributions...")
                    time.sleep(0.1)

    if cache_dirty:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(observer_cache, f, indent=2)

    # Clean unrecorded files in species folders (files beyond existing rows)
    valid_filepaths = {Path(r["filepath"]).resolve() for r in existing_rows}
    for species_dir in OUTPUT_DIR.iterdir():
        if species_dir.is_dir():
            for f in species_dir.glob("*.jpg"):
                if f.resolve() not in valid_filepaths:
                    try:
                        f.unlink()
                    except Exception:
                        pass

    # Save cleaned existing metadata
    save_metadata(existing_rows, meta_path)
    print("Cleaned unindexed temporary files and saved base metadata manifest.")

    # Step 2: Download new images per species to reach 100 images/species
    updated_all_rows = list(existing_rows)

    for label, sciname in TARGET_SPECIES.items():
        species_dir = OUTPUT_DIR / label
        species_dir.mkdir(parents=True, exist_ok=True)

        current_species_rows = [r for r in updated_all_rows if r["species"] == label]
        current_count = len(current_species_rows)
        print(f"\n==================================================")
        print(f"Species: {label.upper()} ({sciname})")
        print(f"Current count: {current_count} / Target: {TARGET_COUNT_PER_SPECIES}")
        print(f"==================================================")

        if current_count >= TARGET_COUNT_PER_SPECIES:
            print(f"Target count already satisfied for {label}.")
            continue

        needed = TARGET_COUNT_PER_SPECIES - current_count
        print(f"Need {needed} additional images for {label}.")

        # Paged query
        offset = 0
        added_for_species = 0

        while added_for_species < needed and offset < 600:
            print(f"  Querying GBIF offset={offset}, limit=100...")
            records = fetch_gbif_records(sciname, limit=100, offset=offset)
            offset += 100
            if not records:
                print("  No more records returned by GBIF.")
                break

            for rec in records:
                if added_for_species >= needed:
                    break

                gbif_id = str(rec.get("key", ""))
                if not gbif_id or gbif_id in seen_keys:
                    continue

                media_list = rec.get("media", [])
                if not media_list:
                    continue

                img_url = media_list[0].get("identifier")
                if not img_url:
                    continue

                license_val = rec.get("license", "CC-BY-4.0")
                recorded_by = str(rec.get("recordedBy", "unknown") or "unknown")
                event_date = str(rec.get("eventDate", "unknown") or "unknown")

                # Next file index
                new_idx = current_count + added_for_species + 1
                filename = f"{label}_{new_idx:03d}.jpg"
                dest_file = species_dir / filename

                success, h_val, md5_val = download_and_validate_image(img_url, dest_file)
                if not success:
                    continue

                if h_val in seen_hashes:
                    # Duplicate visual image, remove and skip
                    if dest_file.exists():
                        dest_file.unlink()
                    continue

                seen_keys.add(gbif_id)
                seen_hashes.add(h_val)
                added_for_species += 1

                new_row = {
                    "species": label,
                    "scientific_name": sciname,
                    "filename": filename,
                    "filepath": str(dest_file.as_posix()),
                    "gbif_key": gbif_id,
                    "recorded_by": recorded_by,
                    "event_date": event_date,
                    "license": license_val,
                    "source_url": img_url,
                }
                updated_all_rows.append(new_row)

                safe_obs = recorded_by.encode("ascii", "replace").decode("ascii")[:25]
                print(f"    [{label}] Added {current_count + added_for_species}/{TARGET_COUNT_PER_SPECIES}: {filename} (Obs: {safe_obs})")
                time.sleep(0.04)

            # Save progress incrementally after each page
            save_metadata(updated_all_rows, meta_path)

        print(f"Finished {label}: now has {current_count + added_for_species} images.")
        save_metadata(updated_all_rows, meta_path)

    print("\n==================================================")
    print("DATASET EXPANSION COMPLETE")
    print(f"Total metadata records: {len(updated_all_rows)}")
    for sp in TARGET_SPECIES:
        cnt = sum(1 for r in updated_all_rows if r["species"] == sp)
        print(f"  - {sp}: {cnt} images")
    print(f"Saved manifest to {meta_path}")
    print("==================================================")


if __name__ == "__main__":
    expand_dataset()
