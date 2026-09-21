*This is a machine learning model which predicts a distribution of 16 colors given a word as input.* 

# Selection of colors
The colors are in CIELAB color space instead of RGB, because the CIELAB color space is perceptually uniform. This means that distances between two color points corresponds to the amount of visual difference perceived by the human eye. The 16 colors were chosen in a way that the eucledian distance between the colors is maximized (see `distinct_colors.py`). The script generates a palette of visually distinct colors by solving a maximin optimization problem: it tries to maximize the minimum distance between any two selected colors.
Finding the globally optimal set of colors is a difficult non-cl Generated Palette

## Final Generated Palette

| Swatch | Name | CIELAB |
|---|---|---|
| <img src="https://singlecolorimage.com/get/000000/20x20" width="20" height="20" /> | Black | `0.00, 0.00, 0.00` |
| <img src="https://singlecolorimage.com/get/FFFFFF/20x20" width="20" height="20" /> | White | `100.00, -0.00, 0.00` |
| <img src="https://singlecolorimage.com/get/1080DF/20x20" width="20" height="20" /> | Azure Blue | `52.81, 6.85, -56.17` |
| <img src="https://singlecolorimage.com/get/EF0000/20x20" width="20" height="20" /> | Bright Red | `49.93, 76.26, 63.99` |
| <img src="https://singlecolorimage.com/get/008F00/20x20" width="20" height="20" /> | Forest Green | `51.43, -56.02, 54.07` |
| <img src="https://singlecolorimage.com/get/FF00EF/20x20" width="20" height="20" /> | Magenta | `59.43, 96.07, -53.34` |
| <img src="https://singlecolorimage.com/get/CF9F00/20x20" width="20" height="20" /> | Mustard Yellow | `68.07, 6.41, 71.74` |
| <img src="https://singlecolorimage.com/get/4000AF/20x20" width="20" height="20" /> | Deep Violet | `24.27, 61.27, -74.80` |
| <img src="https://singlecolorimage.com/get/705010/20x20" width="20" height="20" /> | Golden Brown | `36.41, 7.33, 39.77` |
| <img src="https://singlecolorimage.com/get/9FFF00/20x20" width="20" height="20" /> | Lime Green | `91.18, -58.22, 87.37` |
| <img src="https://singlecolorimage.com/get/00BFBF/20x20" width="20" height="20" /> | Turquoise | `70.19, -38.70, -11.37` |
| <img src="https://singlecolorimage.com/get/006060/20x20" width="20" height="20" /> | Dark Teal | `36.38, -23.52, -6.91` |
| <img src="https://singlecolorimage.com/get/706070/20x20" width="20" height="20" /> | Mauve Gray | `42.73, 9.56, -6.65` |
| <img src="https://singlecolorimage.com/get/FF9F9F/20x20" width="20" height="20" /> | Salmon Pink | `75.18, 35.66, 15.16` |
| <img src="https://singlecolorimage.com/get/600000/20x20" width="20" height="20" /> | Maroon | `17.86, 39.17, 27.56` |
| <img src="https://singlecolorimage.com/get/BFAFEF/20x20" width="20" height="20" /> | Lavender | `74.80, 18.56, -29.83` |
