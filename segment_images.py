"""Create SAM 2.1 object masks and an offline HTML inspection gallery.

python segment_images.py --limit 24
python segment_images.py --limit 0  # all eligible boxes, resumable
"""

import argparse
from collections import defaultdict
import csv
from html import escape
import json
import os
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


DEFAULT_MODEL = "facebook/sam2.1-hiera-tiny"


def load_objects(dataset, seed, categories):
    with (dataset / "samples.csv").open(newline="", encoding="utf-8") as file:
        samples = {(row["image_id"], row["label_id"]): row for row in csv.DictReader(file)}
    grouped = defaultdict(list)
    skipped = 0
    with (dataset / "boxes.csv").open(newline="", encoding="utf-8") as file:
        for index, row in enumerate(csv.DictReader(file)):
            sample = samples.get((row["ImageID"], row["LabelName"]))
            if sample is None:
                continue
            if categories and sample["category"].casefold() not in categories:
                continue
            # Group, depicted, and inside-another-object boxes are unsuitable
            # for individual visible-object color extraction.
            if float(row["Confidence"]) != 1 or any(
                row[field] == "1" for field in ("IsGroupOf", "IsDepiction", "IsInside")
            ):
                skipped += 1
                continue
            box = [float(row[key]) for key in ("XMin", "YMin", "XMax", "YMax")]
            if not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
                skipped += 1
                continue
            grouped[sample["category"]].append({
                **sample, "id": f'{row["ImageID"]}_box{index:05d}',
                "box_normalized": box, "occluded": row["IsOccluded"] == "1",
                "truncated": row["IsTruncated"] == "1",
            })
    if categories:
        missing = categories - {name.casefold() for name in grouped}
        if missing:
            raise ValueError(f"No eligible objects for categories: {sorted(missing)}")
    # Round-robin categories so a small preview is diverse, not all accordions.
    rng = random.Random(seed)
    names = sorted(grouped)
    rng.shuffle(names)
    for items in grouped.values():
        rng.shuffle(items)
    ordered = []
    for index in range(max(map(len, grouped.values()), default=0)):
        for name in names:
            if index < len(grouped[name]):
                ordered.append(grouped[name][index])
    return ordered, skipped


def mask_metrics(mask, box, score):
    height, width = mask.shape
    x1, y1 = np.floor(box[:2]).astype(int)
    x2, y2 = np.ceil(box[2:]).astype(int)
    x1, x2 = np.clip([x1, x2], 0, width)
    y1, y2 = np.clip([y1, y2], 0, height)
    area = int(mask.sum())
    inside = int(mask[y1:y2, x1:x2].sum())
    outside_fraction = (area - inside) / max(area, 1)
    box_coverage = inside / max((x2 - x1) * (y2 - y1), 1)
    flags = []
    if area == 0:
        flags.append("empty mask")
    if score < 0.8:
        flags.append("low predicted quality")
    if outside_fraction > 0.1:
        flags.append("mask extends beyond prompt box")
    if box_coverage < 0.05:
        flags.append("very small foreground")
    return {"foreground_pixels": area, "outside_box_fraction": outside_fraction,
            "box_coverage": box_coverage, "flags": flags}


def save_previews(image, mask, box, prefix):
    """Downsample only gallery assets; saved masks retain original resolution."""
    preview = image.copy()
    preview.thumbnail((480, 360), Image.Resampling.LANCZOS)
    binary = Image.fromarray(mask.astype(np.uint8) * 255)
    small_mask = binary.resize(preview.size, Image.Resampling.NEAREST)
    scale = np.array([preview.width / image.width, preview.height / image.height] * 2)
    rectangle = tuple(np.asarray(box) * scale)
    original = preview.copy()
    ImageDraw.Draw(original).rectangle(rectangle, outline="#ffe600", width=2)
    original.save(f"{prefix}_original.jpg", quality=90)
    tint = Image.blend(preview, Image.new("RGB", preview.size, "#00e5b0"), 0.45)
    overlay = Image.composite(tint, preview, small_mask)
    boundary = np.asarray(small_mask) != np.asarray(small_mask.filter(ImageFilter.MinFilter(3)))
    pixels = np.array(overlay)
    pixels[boundary] = [255, 255, 0]
    overlay = Image.fromarray(pixels)
    ImageDraw.Draw(overlay).rectangle(rectangle, outline="#ffe600", width=2)
    overlay.save(f"{prefix}_overlay.jpg", quality=90)
    foreground = preview.convert("RGBA")
    foreground.putalpha(small_mask)
    foreground.save(f"{prefix}_foreground.png")


def write_gallery(output, results, dataset, limit):
    cards = []
    # Show suspicious results first to make inspection useful.
    ordered = sorted(results, key=lambda r: (not bool(r["flags"]), r["predicted_iou"]))
    for row in ordered[:limit]:
        key = row["id"]
        panels = []
        for suffix, title, extension in [
            ("original", "Original + prompt box", "jpg"),
            ("overlay", "Predicted mask overlay", "jpg"),
            ("foreground", "Extracted foreground", "png"),
        ]:
            src = f"previews/{key}_{suffix}.{extension}"
            panels.append(f'<figure><a href="{src}" target="_blank"><img loading="lazy" '
                          f'src="{src}" alt="{escape(title)}"></a><figcaption>{title}</figcaption></figure>')
        flags = "; ".join(row["flags"]) or "No automatic flags"
        original = escape(Path(os.path.relpath(dataset / row["path"], output)).as_posix(), quote=True)
        cards.append(
            f'<article data-name="{escape(row["category"].casefold(), quote=True)}" '
            f'data-flagged="{int(bool(row["flags"]))}"><h2>{escape(row["category"])}</h2>'
            f'<p class="meta">{key} · model quality estimate: {row["predicted_iou"]:.3f} · '
            f'{row["foreground_pixels"]:,} foreground pixels</p><div class="panels">'
            + "".join(panels) + f'</div><p>{escape(flags)}</p>'
            f'<p><a href="masks/{key}.png" target="_blank">Full-resolution binary mask</a> · '
            f'<a href="{original}" target="_blank">Full-resolution original</a> · '
            f'<a href="records/{key}.json" target="_blank">Metadata</a></p></article>'
        )
    html = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Segmentation inspection</title>
<style>
body{font:16px system-ui,sans-serif;background:#111827;color:#e5e7eb;max-width:1500px;margin:30px auto;padding:0 20px}
h1{margin-bottom:8px}h2{margin:0}a{color:#67e8f9}header p{max-width:1000px;line-height:1.5}
article{background:#1f2937;border:1px solid #374151;border-radius:12px;padding:20px;margin:20px 0}
.panels{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0}
img{width:100%;height:300px;object-fit:contain;background:repeating-conic-gradient(#374151 0% 25%,#4b5563 0% 50%) 0/20px 20px}
figcaption{padding:8px 0}.meta{color:#cbd5e1;font-size:13px;overflow-wrap:anywhere}
input[type=search]{padding:10px;border-radius:6px;margin:8px 16px 8px 0;width:240px}
[hidden]{display:none!important}@media(max-width:750px){.panels{grid-template-columns:1fr}img{height:auto;max-height:400px}}
</style></head><body><header><h1>Segmentation inspection</h1>
<p>Box-guided SAM 2.1 predictions. Check missing object parts, background leakage, and holes.
Yellow rectangles are prompts, not masks. Quality scores are model estimates, not measured accuracy.
Suspicious masks appear first. Gallery thumbnails are resized; binary masks use original image dimensions.</p>
'''
    html += f'<p>Showing {len(cards)} of {len(results)} generated masks.</p>'
    html += '''<input id="search" type="search" placeholder="Filter by category" aria-label="Filter by category">
<label><input id="flagged" type="checkbox"> Only flagged results</label></header><main>'''
    html += "".join(cards)
    html += '''</main><script>
const search=document.querySelector('#search'), flagged=document.querySelector('#flagged');
function filter(){for(const card of document.querySelectorAll('article')){
card.hidden=!card.dataset.name.includes(search.value.toLowerCase()) || (flagged.checked && card.dataset.flagged==='0');}}
search.addEventListener('input',filter);flagged.addEventListener('change',filter);
</script></body></html>'''
    (output / "index.html").write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("/mnt/d/color_guesser"))
    parser.add_argument("--output", type=Path, help="Default: DATASET/segmentation")
    parser.add_argument("--limit", type=int, default=24, help="Object masks to produce; 0 means all")
    parser.add_argument("--category", action="append", default=[], help="Exact name; repeatable")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face ID or local model directory")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=4, help="Box prompts per inference batch")
    parser.add_argument("--gallery-limit", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing masks")
    args = parser.parse_args()
    if args.limit < 0 or min(args.batch_size, args.gallery_limit) < 1:
        parser.error("Limit must be nonnegative; batch size and gallery limit must be positive")
    dataset = args.dataset.resolve()
    output = (args.output or dataset / "segmentation").resolve()
    objects, skipped = load_objects(dataset, args.seed, {c.casefold() for c in args.category})
    eligible = len(objects)
    if args.limit:
        objects = objects[:args.limit]
    if not objects:
        parser.error("No eligible objects found")
    for directory in ("masks", "previews", "records"):
        (output / directory).mkdir(parents=True, exist_ok=True)

    results, pending, errors = [], defaultdict(list), []
    for obj in objects:
        record = output / "records" / f'{obj["id"]}.json'
        signature = {"model": args.model, "box": obj["box_normalized"],
                     "source_bytes": (dataset / obj["path"]).stat().st_size, "pipeline_version": 1}
        obj["signature"] = signature
        artifacts = [output / "masks" / f'{obj["id"]}.png'] + [
            output / "previews" / f'{obj["id"]}_{suffix}.{extension}'
            for suffix, extension in [("original", "jpg"), ("overlay", "jpg"), ("foreground", "png")]
        ]
        if record.exists() and not args.overwrite and all(path.exists() for path in artifacts):
            saved = json.loads(record.read_text())
            if saved.get("signature") == signature:
                results.append(saved)
                continue
        pending[obj["path"]].append(obj)

    print(f"Selected {len(objects)} of {eligible} eligible objects; {skipped} boxes excluded. "
          f"Reusing {len(results)} saved masks.", flush=True)
    device = "not needed (cached results)"
    if pending:
        import torch
        from transformers import Sam2Model, Sam2Processor

        device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use --device cpu")
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        print(f"Loading {args.model} on {device}...", flush=True)
        processor = Sam2Processor.from_pretrained(args.model)
        model = Sam2Model.from_pretrained(args.model, use_safetensors=True).to(device).eval()

        for image_path, tasks in pending.items():
            try:
                # Match the dataset's stored pixel orientation; do not silently
                # rotate via EXIF, which would misalign the provided boxes.
                with Image.open(dataset / image_path) as source:
                    image = source.convert("RGB")
                embeddings = None
                for start in range(0, len(tasks), args.batch_size):
                    batch = tasks[start:start + args.batch_size]
                    boxes = [np.asarray(obj["box_normalized"]) *
                             [image.width, image.height, image.width, image.height] for obj in batch]
                    inputs = processor(images=image, input_boxes=[[b.tolist() for b in boxes]],
                                       return_tensors="pt").to(device)
                    if embeddings is not None:
                        inputs.pop("pixel_values")
                        inputs["image_embeddings"] = embeddings
                    with torch.inference_mode():
                        predictions = model(**inputs, multimask_output=True)
                    embeddings = predictions.image_embeddings
                    scores = predictions.iou_scores[0].float().cpu().numpy()
                    for index, (obj, box) in enumerate(zip(batch, boxes)):
                        best = int(np.argmax(scores[index]))
                        # Upsample only the best candidate, one object at a time.
                        logits = predictions.pred_masks[:, index:index + 1, best:best + 1].float().cpu()
                        mask = processor.post_process_masks(
                            logits, inputs["original_sizes"].cpu(), binarize=True,
                        )[0][0, 0].numpy().astype(bool)
                        if mask.shape != (image.height, image.width):
                            raise RuntimeError(f"Mask dimensions do not match source: {mask.shape}")
                        score = float(scores[index, best])
                        if not np.isfinite(score):
                            raise RuntimeError("Non-finite model quality score")
                        metrics = mask_metrics(mask, box, score)
                        if obj["occluded"]:
                            metrics["flags"].append("annotated occlusion")
                        if obj["truncated"]:
                            metrics["flags"].append("annotated truncation")
                        row = {**obj, **metrics, "predicted_iou": score, "box_pixels": box.tolist(),
                               "width": image.width, "height": image.height,
                               "mask_path": f'masks/{obj["id"]}.png',
                               "model_revision": getattr(model.config, "_commit_hash", None)}
                        Image.fromarray(mask.astype(np.uint8) * 255).save(output / row["mask_path"])
                        save_previews(image, mask, box, output / "previews" / obj["id"])
                        (output / "records" / f'{obj["id"]}.json').write_text(json.dumps(row, indent=2) + "\n")
                        results.append(row)
                print(f'{len(results)}/{len(objects)} masks saved: {tasks[0]["category"]}', flush=True)
            except Exception as error:
                errors.append({"path": image_path, "error": str(error)})
                print(f"ERROR {image_path}: {error}", flush=True)

    write_gallery(output, results, dataset, args.gallery_limit)
    (output / "manifest.json").write_text(json.dumps(results, indent=2) + "\n")
    summary = {"model": args.model, "device": device, "selected_objects": len(objects),
               "saved_masks": len(results), "flagged_masks": sum(bool(r["flags"]) for r in results),
               "excluded_boxes": skipped, "errors": errors}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Open gallery: {output / 'index.html'}")
    if errors:
        raise SystemExit("Some objects failed; see summary.json. Rerun to resume.")


if __name__ == "__main__":
    main()
