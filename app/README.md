# Road Damage Detection API

YOLO11n (exported to ONNX, served with ONNX Runtime) trained on **RDD2022** (China_Drone, China_MotorBike, Czech, India, Japan, United_States; Norway excluded)
to detect four road-surface defect classes, plus a severity score per defect and a Road Condition Index per image.

| ID | RDD code | Class |
|---|---|---|
| 0 | D00 | longitudinal_crack |
| 1 | D10 | transverse_crack |
| 2 | D20 | alligator_crack |
| 3 | D40 | pothole |

## Endpoints

| Method | Path | Returns |
|---|---|---|
| GET | `/` | Upload page |
| GET | `/docs` | Swagger UI |
| GET | `/health` | Model + class info |
| GET | `/metrics` | Validation mAP@0.5, per-class and per-country results |
| POST | `/predict` | JSON: detections, severity, Road Condition Index |
| POST | `/predict/image` | Annotated JPEG (boxes coloured by severity), summary in `X-Road-Condition` header |

```bash
curl -F "file=@road.jpg" "https://road-damage-detection.onrender.com/predict?conf=0.25"
```

## Severity scoring

Per detection, `severity = 100 × class_weight × extent × (0.5 + 0.5 × confidence)`, where
class weights are pothole 1.0, alligator 0.85, transverse 0.55, longitudinal 0.45 and `extent`
is the log-scaled fraction of the image the box covers. Levels: low < 35 ≤ medium < 65 ≤ high.

Road Condition Index (0–100, higher is better) = `100 × Π(1 − severity_i / 150)`,
rated good ≥ 85, fair ≥ 60, poor ≥ 35, else critical.

Weights are heuristic (not calibrated against field PCI surveys).

## Run locally with Docker

```bash
docker build -t road-damage .
docker run -p 7860:7860 road-damage
```

Dataset: Arya et al., *RDD2022: A multi-national image dataset for automatic road damage detection*, CC BY-SA 4.0.
