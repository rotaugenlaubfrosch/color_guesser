"""Download a reproducible Open Images subset. Requires curl and Pillow.

Run: python download_open_images.py --output /mnt/d/color_guesser
The validation split's published V5 boxes are also used by later releases.
Bounding boxes are normalized coordinates, not segmentation masks.
"""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import random
import subprocess
import shutil
import time

from PIL import Image


SOURCES = {
    "classes.csv": "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv",
    "validation-boxes.csv": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
    "validation-images.csv": "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
}

# Published content lengths, checked against the servers before this download.
METADATA_BYTES = {"validation-boxes.csv": 25105048, "validation-images.csv": 15245485}


def download_metadata(url, path):
    size = METADATA_BYTES.get(path.name)
    if not size:
        return download(url, path)
    if path.exists() and path.stat().st_size == size:
        return
    chunks = path.parent / (path.name + ".chunks")
    chunks.mkdir(exist_ok=True)
    chunk_size = 500_000

    def fetch_chunk(start):
        end = min(start + chunk_size, size) - 1
        chunk = chunks / str(start)
        if chunk.exists() and chunk.stat().st_size == end - start + 1:
            return chunk
        subprocess.run(
            ["curl", "-4", "--fail", "--location", "--silent", "--show-error",
             "--retry", "3", "--connect-timeout", "15", "--max-time", "600",
             "--range", f"{start}-{end}", "--output", str(chunk), url], check=True,
        )
        if chunk.stat().st_size != end - start + 1:
            raise RuntimeError(f"Incorrect chunk size: {chunk}")
        return chunk

    with ThreadPoolExecutor(max_workers=32) as pool:
        parts = list(pool.map(fetch_chunk, range(0, size, chunk_size)))
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as target:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, target)
    temporary.replace(path)
    for part in parts:
        part.unlink()
    chunks.rmdir()


def download(url, path):
    if path.exists() and path.stat().st_size:
        return
    temporary = path.with_suffix(path.suffix + ".part")
    # Restart curl on each attempt so resume begins at the newly downloaded
    # offset, rather than rewinding to the offset from the first attempt.
    for attempt in range(8):
        try:
            subprocess.run(
                ["curl", "-4", "--fail", "--location", "--silent", "--show-error",
                 "--connect-timeout", "15", "--max-time", "180",
                 "--continue-at", "-", "--output", str(temporary), url],
                check=True,
            )
            break
        except subprocess.CalledProcessError:
            if attempt == 7:
                raise
            time.sleep(2)
    temporary.replace(path)


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/mnt/d/color_guesser"))
    parser.add_argument("--categories", type=int, default=300)
    parser.add_argument("--per-category", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=48)
    args = parser.parse_args()
    if min(args.categories, args.per_category, args.workers) < 1:
        parser.error("Counts must be positive")
    root = args.output
    metadata, images = root / "metadata", root / "images"
    metadata.mkdir(parents=True, exist_ok=True)
    images.mkdir(exist_ok=True)
    for name, url in SOURCES.items():
        print(f"Downloading metadata: {name}", flush=True)
        download_metadata(url, metadata / name)

    with (metadata / "classes.csv").open(encoding="utf-8") as file:
        names = dict(csv.reader(file))
    with (metadata / "validation-images.csv").open(encoding="utf-8") as file:
        attribution = {row["ImageID"]: row for row in csv.DictReader(file)}
    with (metadata / "validation-boxes.csv").open(encoding="utf-8") as file:
        reader = csv.DictReader(file)
        box_fields = reader.fieldnames
        boxes = list(reader)

    candidates = defaultdict(set)
    for row in boxes:
        # Prefer individually localized objects rather than groups or depictions.
        if (row["ImageID"] in attribution and row["IsGroupOf"] == "0"
                and row["IsDepiction"] == "0" and row["Confidence"] == "1"):
            candidates[row["LabelName"]].add(row["ImageID"])
    eligible = sorted(label for label, ids in candidates.items()
                      if len(ids) >= args.per_category)
    print(f"Eligible categories: {len(eligible)}", flush=True)
    if len(eligible) < args.categories:
        raise RuntimeError(f"Only {len(eligible)} eligible categories; need {args.categories}")
    rng = random.Random(args.seed)
    labels = sorted(rng.sample(eligible, args.categories), key=lambda label: names[label])
    selections = []
    categories = []
    for label in labels:
        chosen = rng.sample(sorted(candidates[label]), args.per_category)
        categories.append({"label_id": label, "name": names[label], "image_ids": chosen})
        for image_id in chosen:
            selections.append({"label_id": label, "category": names[label],
                               "image_id": image_id, "path": f"images/{image_id}.jpg"})
    image_ids = sorted({row["image_id"] for row in selections})
    (root / "categories.json").write_text(json.dumps(categories, indent=2) + "\n")
    write_csv(root / "samples.csv", selections, ["label_id", "category", "image_id", "path"])
    selected_pairs = {(row["image_id"], row["label_id"]) for row in selections}
    selected_boxes = [row for row in boxes
                      if (row["ImageID"], row["LabelName"]) in selected_pairs]
    write_csv(root / "boxes.csv", selected_boxes, box_fields)
    records = [dict(attribution[i], DownloadURL=
                    f"https://open-images-dataset.s3.amazonaws.com/validation/{i}.jpg")
               for i in image_ids]
    write_csv(root / "attribution.csv", records, list(records[0]))

    print(f"Downloading and verifying {len(image_ids)} unique images for "
          f"{len(labels)} categories", flush=True)

    def fetch(image_id):
        path = images / f"{image_id}.jpg"
        download(f"https://open-images-dataset.s3.amazonaws.com/validation/{image_id}.jpg", path)
        with Image.open(path) as image:
            image.load()
        return path.stat().st_size

    failures, total_bytes = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(fetch, image_id): image_id for image_id in image_ids}
        for done, job in enumerate(as_completed(jobs), 1):
            try:
                total_bytes += job.result()
            except Exception as error:
                failures.append({"image_id": jobs[job], "error": str(error)})
            if done % 100 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} processed; failures: {len(failures)}", flush=True)

    summary = {
        "dataset": "Open Images", "split": "validation", "seed": args.seed,
        "categories": len(labels), "images_per_category": args.per_category,
        "category_image_pairs": len(selections), "unique_images": len(image_ids),
        "verified_images": len(image_ids) - len(failures), "image_bytes": total_bytes,
        "bounding_boxes": len(selected_boxes), "metadata_sources": SOURCES,
        "selection": "Uniform random eligible categories; uniform random images per category. "
                     "Eligibility requires a non-group, non-depiction box with confidence 1. "
                     "boxes.csv retains all boxes for the selected image/category pairs.",
        "failures": failures,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (root / "README.md").write_text(
        "# Open Images color dataset subset\n\n"
        "- `images/`: original downloaded JPEGs, stored once per image ID.\n"
        "- `samples.csv`: exactly the requested image assignments per category.\n"
        "- `categories.json`: category names, IDs, and assigned image IDs.\n"
        "- `boxes.csv`: normalized bounding boxes for selected image/category pairs.\n"
        "- `attribution.csv`: original URLs, author, title, license, and download URL.\n"
        "- `metadata/`: source CSVs used for reproducible selection.\n"
        "- `summary.json`: counts, selection method, URLs, and any download failures.\n\n"
        "Images are listed by Open Images as CC BY 2.0; see per-image License and Author "
        "in attribution.csv. Verify individual licenses when redistributing. "
        "Annotations are CC BY 4.0. Preserve attribution.\n"
        "https://storage.googleapis.com/openimages/web/factsfigures_v7.html\n\n"
        "Bounding boxes are not segmentation masks. The selection is object-focused "
        "and does not supply brand/country association labels. Split concepts yourself "
        "for model evaluation; the source split name does not define your task's splits.\n"
    )
    print(json.dumps(summary, indent=2), flush=True)
    if failures:
        raise SystemExit("Some downloads failed. Rerun the same command to retry missing files.")


if __name__ == "__main__":
    main()
