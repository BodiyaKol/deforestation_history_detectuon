# deforestation_history_detectuon

Here we implement the algorithm to detect anomalies in historical data of a specific area, which are likely caused by deforestation.

---

## Demo
Normal deforestation
![Program Demo](output/video/deforestation.gif)

Z-score for each pixel
![Program Demo](output/video/deforestation_overlay.gif)

---

## Testing

To run tests:

1. Generate test images: `python tests/generate_test_images.py --all`
2. Run unit tests: `python tests/test_pipeline.py --unit`
3. Run integration tests: `python tests/test_pipeline.py --integration`

Results are saved in `tests/test_results/regression/` with `summary.txt` and individual `metrics.txt` files for each scenario.

---

### Video 1 (Kolodchak Bohdan)
[![Video 1](https://img.youtube.com/vi/VIDEO_ID_1/0.jpg)](https://www.youtube.com/watch?v=VIDEO_ID_1)

### Video 2 (Pasternak Yullia)
[![Video 2](https://img.youtube.com/vi/VIDEO_ID_2/0.jpg)](https://www.youtube.com/watch?v=VIDEO_ID_2)

### Video 3 (Prokopets Maxym)
[![Video 3](https://img.youtube.com/vi/VIDEO_ID_3/0.jpg)](https://www.youtube.com/watch?v=VIDEO_ID_3)

---
