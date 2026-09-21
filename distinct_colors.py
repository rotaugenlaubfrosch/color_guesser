"""Find 16 displayable colors with a large minimum pairwise CIEDE2000 distance.

Install: pip install numpy scikit-image
Run:     python distinct_colors.py
Output:  colors.html beside this script

Greedy farthest-point sampling with random restarts approximates the maximin
objective; it does not guarantee the globally optimal palette. Candidates are
an sRGB grid converted to CIELAB (D65), so all results are displayable.
Pure black and pure white are always included; the remaining colors are selected
to maximize separation, including their distances from black and white.
"""

from pathlib import Path

import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab


# Human-readable labels for the default generated palette.
COLOR_NAMES = {
    "#000000": "Black",
    "#FFFFFF": "White",
    "#1080DF": "Azure Blue",
    "#EF0000": "Bright Red",
    "#008F00": "Forest Green",
    "#FF00EF": "Magenta",
    "#CF9F00": "Mustard Yellow",
    "#4000AF": "Deep Violet",
    "#705010": "Golden Brown",
    "#9FFF00": "Lime Green",
    "#00BFBF": "Turquoise",
    "#006060": "Dark Teal",
    "#706070": "Mauve Gray",
    "#FF9F9F": "Salmon Pink",
    "#600000": "Maroon",
    "#BFAFEF": "Lavender",
}


def find_colors(count=16, grid_size=17, restarts=16, seed=42):
    if count < 2 or grid_size < 2 or restarts < 1:
        raise ValueError("count/grid_size must be >= 2, and restarts >= 1")

    levels = np.rint(np.linspace(0, 255, grid_size)).astype(np.uint8)
    rgb = np.unique(
        np.stack(np.meshgrid(levels, levels, levels), axis=-1).reshape(-1, 3),
        axis=0,
    )
    if count > len(rgb):
        raise ValueError("Not enough candidate colors")
    lab = rgb2lab(rgb / 255.0, illuminant="D65")
    rng = np.random.default_rng(seed)
    best_indices, best_score = None, -np.inf
    anchors = [0, len(rgb) - 1]  # np.unique sorts RGB: black first, white last.

    for _ in range(restarts):
        selected = anchors.copy()
        nearest = np.minimum(
            deltaE_ciede2000(lab, lab[anchors[0]][None, :]),
            deltaE_ciede2000(lab, lab[anchors[1]][None, :]),
        )
        if count > 2:
            selected.append(int(rng.integers(1, len(rgb) - 1)))
        while len(selected) < count:
            distances = deltaE_ciede2000(lab, lab[selected[-1]][None, :])
            nearest = np.minimum(nearest, distances)
            nearest[selected] = -np.inf
            selected.append(int(np.argmax(nearest)))

        palette = lab[selected]
        pairwise = deltaE_ciede2000(palette[:, None, :], palette[None, :, :])
        np.fill_diagonal(pairwise, np.inf)
        score = pairwise.min()
        if score > best_score:
            best_indices, best_score = selected, score

    return rgb[best_indices], lab[best_indices], float(best_score)


if __name__ == "__main__":
    rgb, lab, score = find_colors()
    print(f"Minimum pairwise CIEDE2000 distance: {score:.2f}\n")
    print("Hex       L*       a*       b*")
    cards = []
    for color, (lightness, a, b) in zip(rgb, lab):
        hex_color = "#" + "".join(f"{int(channel):02X}" for channel in color)
        print(f"{hex_color}  {lightness:7.2f}  {a:7.2f}  {b:7.2f}")
        cards.append(
            f'<article><div class="swatch" style="background:{hex_color}"></div>'
            f'<strong>{COLOR_NAMES.get(hex_color, "Custom Color")}</strong>'
            f'<p>{hex_color}</p>'
            f'<p>Lab: {lightness:.2f}, {a:.2f}, {b:.2f}</p></article>'
        )

    output = Path(__file__).with_name("colors.html")
    output.write_text(
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>Distinct color palette</title><style>'
        'body{font-family:system-ui,sans-serif;max-width:1000px;margin:40px auto;'
        'padding:0 20px;background:#f4f4f4;color:#222}'
        '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px}'
        'article{background:white;border:1px solid #ddd;border-radius:10px;overflow:hidden}'
        '.swatch{height:150px;border-bottom:1px solid #ddd}'
        'strong{display:block;margin:14px 14px 0}article p{margin:8px 14px 16px;font-size:13px}'
        '</style></head><body>'
        f'<h1>{len(rgb)} distinct colors</h1>'
        f'<p>Minimum pairwise CIEDE2000 distance: {score:.2f}</p>'
        '<main class="grid">' + ''.join(cards) + '</main></body></html>\n',
        encoding="utf-8",
    )
    print(f"\nSaved palette preview to {output}")
