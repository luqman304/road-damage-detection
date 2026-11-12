"""FastAPI service for road-damage detection + severity scoring."""
from __future__ import annotations
import io, json, os, pathlib, time

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, Response
from PIL import Image, ImageDraw, ImageFont

from detector import OnnxYOLO
from severity import detection_severity, road_condition

ROOT = pathlib.Path(__file__).parent
MODEL_PATH = os.getenv("MODEL_PATH", str(ROOT / "weights" / "best.onnx"))
METRICS_PATH = ROOT / "weights" / "metrics.json"
MAX_BYTES = 15 * 1024 * 1024

model = OnnxYOLO(MODEL_PATH)
NAMES = model.names
COLORS = {"high": (220, 38, 38), "medium": (245, 158, 11), "low": (34, 197, 94)}

app = FastAPI(
    title="Road Damage Detection API",
    description="YOLO11n (ONNX Runtime) trained on RDD2022 (D00 longitudinal crack, D10 transverse crack, "
                "D20 alligator crack, D40 pothole) with per-defect severity scoring and a "
                "Road Condition Index.",
    version="1.0.0",
)


def _load(data: bytes) -> Image.Image:
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image larger than 15 MB")
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "File is not a valid image")


def _analyse(img: Image.Image, conf: float, iou: float) -> dict:
    t0 = time.perf_counter()
    w, h = img.size
    dets = []
    for box, c, k in model.predict(img, conf=conf, iou=iou):
        name = NAMES[int(k)]
        d = {"class_id": int(k), "class_name": name, "confidence": round(c, 4),
             "box_xyxy": [round(v, 1) for v in box]}
        d.update(detection_severity(name, c, box, w, h))
        dets.append(d)
    dets.sort(key=lambda d: -d["severity_score"])
    counts = {}
    for d in dets:
        counts[d["class_name"]] = counts.get(d["class_name"], 0) + 1
    return {
        "image_size": [w, h],
        "num_detections": len(dets),
        "counts": counts,
        **road_condition([d["severity_score"] for d in dets]),
        "detections": dets,
        "inference_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


def _draw(img: Image.Image, result: dict) -> bytes:
    img = img.copy()
    draw = ImageDraw.Draw(img)
    lw = max(2, round(min(img.size) / 250))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(12, lw * 6))
    except OSError:
        font = ImageFont.load_default()
    for d in result["detections"]:
        col = COLORS[d["severity_level"]]
        x1, y1, x2, y2 = d["box_xyxy"]
        draw.rectangle([x1, y1, x2, y2], outline=col, width=lw)
        label = f'{d["class_name"]} {d["confidence"]:.2f} | sev {d["severity_score"]:.0f}'
        tb = draw.textbbox((x1, y1), label, font=font)
        ty = y1 - (tb[3] - tb[1]) - 4 if y1 > (tb[3] - tb[1]) + 4 else y1
        draw.rectangle([x1, ty, x1 + tb[2] - tb[0] + 6, ty + tb[3] - tb[1] + 4], fill=col)
        draw.text((x1 + 3, ty), label, fill=(255, 255, 255), font=font)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


@app.get("/health")
def health():
    return {"status": "ok", "model": pathlib.Path(MODEL_PATH).name, "classes": NAMES}


@app.get("/metrics")
def metrics():
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    raise HTTPException(404, "metrics.json not bundled")


@app.post("/predict")
async def predict(file: UploadFile = File(...),
                  conf: float = Query(0.25, ge=0.01, le=0.99),
                  iou: float = Query(0.5, ge=0.1, le=0.95)):
    """Detect road damage and return detections with severity scores as JSON."""
    return _analyse(_load(await file.read()), conf, iou)


@app.post("/predict/image", responses={200: {"content": {"image/jpeg": {}}}})
async def predict_image(file: UploadFile = File(...),
                        conf: float = Query(0.25, ge=0.01, le=0.99),
                        iou: float = Query(0.5, ge=0.1, le=0.95)):
    """Same as /predict but returns the annotated JPEG (boxes coloured by severity).
    The JSON summary is in the X-Road-Condition header."""
    img = _load(await file.read())
    res = _analyse(img, conf, iou)
    summary = {k: res[k] for k in ("num_detections", "road_condition_index",
                                   "condition_rating", "max_severity_level")}
    return Response(_draw(img, res), media_type="image/jpeg",
                    headers={"X-Road-Condition": json.dumps(summary)})


@app.get("/", response_class=HTMLResponse)
def index():
    return (ROOT / "static" / "index.html").read_text()
