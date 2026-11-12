# Road & Surface Defect Detection (YOLO11 · RDD2022)

End-to-end road-damage detector: dataset preparation, YOLO11 training on RDD2022,
evaluation (mAP@0.5, per-class, per-country), severity scoring, and a Dockerised FastAPI
service deployed on Render.

**Live API:** https://road-damage-detection-le7h.onrender.com  ·  docs: `/docs`  ·  metrics: `/metrics`
(free tier — first request after 15 min idle takes ~1 min to wake)

## Results

_Final validation results are filled in by `scripts/evaluate.py` once training completes — see `results/RESULTS.md`._

## Pipeline

| Step | Script |
|---|---|
| Download 6 RDD2022 countries via HTTP range requests (Norway skipped, 10.6 GB) | `scripts/download_rdd.py` |
| Pascal-VOC → YOLO, keep D00/D10/D20/D40, seeded 85/15 split per country | `scripts/convert_to_yolo.py` |
| Train YOLO11n, 640px, 25 epochs, Apple MPS | `scripts/train.py`, `scripts/resume.py` |
| Evaluate overall / per-class / per-country | `scripts/evaluate.py` |
| Serve: ONNX Runtime + FastAPI + severity scoring | `app/` |

Dataset: 25,694 train / 4,530 val images; 37,261 / 6,516 labelled defects.

## Severity scoring

See [`app/severity.py`](app/severity.py) and [`app/README.md`](app/README.md).

## Reproduce

```bash
python3.11 -m venv .venv && .venv/bin/pip install ultralytics remotezip onnx onnxruntime onnxslim
.venv/bin/python scripts/download_rdd.py data/raw
.venv/bin/python scripts/convert_to_yolo.py data/raw data/rdd2022_yolo
.venv/bin/python scripts/train.py yolo11n.pt 25 rdd_yolo11n
.venv/bin/python scripts/evaluate.py runs/detect/runs/rdd_yolo11n/weights/best.pt
```

Run the API: `cd app && docker build -t road-damage . && docker run -p 7860:7860 road-damage`

Dataset © Arya et al., RDD2022, CC BY-SA 4.0.
