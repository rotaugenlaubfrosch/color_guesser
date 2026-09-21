# Box-guided object segmentation

`segment_images.py` uses **SAM 2.1 Hiera Tiny** to predict an individual object
mask from each Open Images bounding box. It selects the highest-scoring candidate
mask per object. These are predicted masks, not ground-truth annotations.

## Run

Use the prepared environment on this machine:

```bash
.venv-segmentation/bin/python segment_images.py --limit 24
```

For another machine, create a Python environment, install PyTorch/torchvision for
your CPU or CUDA version, then `pip install -r requirements-segmentation.txt`.
The model downloads from Hugging Face on first use. `--model` also accepts a local
Transformers model directory. GPU is used automatically when available.

```bash
# Inspect a larger, reproducibly sampled selection across categories
python segment_images.py --limit 100

# Restrict to named categories (exact names from categories.json)
python segment_images.py --category Apple --category Cat --limit 20

# Process all eligible object boxes; completed matching results are reused
python segment_images.py --limit 0
```

Dataset defaults to `/mnt/d/color_guesser`; override with `--dataset`.
Output defaults to `<dataset>/segmentation`; override with `--output`.
The limit counts **object instances**, not images. One image may have several
objects of the same category. Group, depiction, and inside-object boxes are excluded.
Occluded/truncated instances are retained but flagged for inspection.

## Inspect

Open `D:\color_guesser\segmentation\index.html` in a browser. Each result shows:

1. Original image with the bounding-box prompt.
2. Colored mask overlay.
3. Foreground on a checkerboard background.

Links open the full-resolution original and binary mask. Search by category or
show only flagged results. The gallery shows up to 200 results, with flagged and
lower-scoring masks first; use `--gallery-limit` to change this. Run a broad sample
as well as inspecting flagged results: model scores do not measure actual accuracy.

Check for background leakage, missing limbs/edges, internal holes, or segmentation
of the wrong object. Masks are not clipped to the prompt box or automatically
filled, which could hide mistakes or incorrectly include background.

## Saved data and later histograms

- `masks/*.png`: full source-resolution binary masks, **0 background / 255 foreground**.
- `records/*.json`: category, source image, prompt, model score, flags, provenance.
- `manifest.json`: records for the current invocation's selection.
- `previews/`: resized inspection assets only.
- `summary.json`: counts and processing errors.

Always extract colors from **original RGB pixels selected by the binary mask**:

```python
rgb = np.asarray(Image.open(original_path).convert("RGB"))
mask = np.asarray(Image.open(mask_path)) > 0
foreground_pixels = rgb[mask]
```

Do not compute histograms from overlays or checkerboard/transparent previews.
For an image-level category histogram, union that category's instance masks first
to avoid counting overlapping pixels twice. An empty mask must be excluded.
The gallery flags are heuristics, not automatic acceptance decisions.
