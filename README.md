*This is a machine learning model which predicts a distribution of 16 colors given a word as input.* 

# Selection of colors
The colors are in CIELAB color space instead of RGB, because the CIELAB color space is perceptually uniform. This means that distances between two color points corresponds to the amount of visual difference perceived by the human eye. The 16 colors were chosen in a way that the eucledian distance between the colors is maximized (see `distinct_colors.py`). The script generates a palette of visually distinct colors by solving a maximin optimization problem: it tries to maximize the minimum distance between any two selected colors.
Finding the globally optimal set of colors is a difficult non-convex optimization problem. Instead, the script uses greedy farthest-point sampling with random restarts: starting from black and white, it repeatedly selects the candidate color that is furthest from its closest already-selected color. This resulted in the below selection of colors:

## Final Generated Palette

Minimum pairwise CIEDE2000 distance: **25.94**

| Hex       | Name           | CIELAB                  |
| --------- | -------------- | ----------------------- |
| `#000000` | Black          | `0.00, 0.00, 0.00`      |
| `#FFFFFF` | White          | `100.00, -0.00, 0.00`   |
| `#1080DF` | Azure Blue     | `52.81, 6.85, -56.17`   |
| `#EF0000` | Bright Red     | `49.93, 76.26, 63.99`   |
| `#008F00` | Forest Green   | `51.43, -56.02, 54.07`  |
| `#FF00EF` | Magenta        | `59.43, 96.07, -53.34`  |
| `#CF9F00` | Mustard Yellow | `68.07, 6.41, 71.74`    |
| `#4000AF` | Deep Violet    | `24.27, 61.27, -74.80`  |
| `#705010` | Golden Brown   | `36.41, 7.33, 39.77`    |
| `#9FFF00` | Lime Green     | `91.18, -58.22, 87.37`  |
| `#00BFBF` | Turquoise      | `70.19, -38.70, -11.37` |
| `#006060` | Dark Teal      | `36.38, -23.52, -6.91`  |
| `#706070` | Mauve Gray     | `42.73, 9.56, -6.65`    |
| `#FF9F9F` | Salmon Pink    | `75.18, 35.66, 15.16`   |
| `#600000` | Maroon         | `17.86, 39.17, 27.56`   |
| `#BFAFEF` | Lavender       | `74.80, 18.56, -29.83`  |


 
