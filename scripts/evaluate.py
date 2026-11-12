"""Evaluate trained weights on the RDD2022 validation split.

Writes metrics.json (overall + per-class P, R, mAP@0.5, mAP@0.5:0.95, plus per-country
mAP@0.5) and a markdown table to RESULTS.md.
usage: python scripts/evaluate.py runs/detect/runs/rdd_yolo11n/weights/best.pt
"""
import json, sys, pathlib, shutil, tempfile
from ultralytics import YOLO

W = sys.argv[1]
DATA = pathlib.Path("data/rdd2022_yolo")
OUT = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "results")
OUT.mkdir(exist_ok=True)
model = YOLO(W)


def run(data_yaml, name):
    m = model.val(data=str(data_yaml), imgsz=640, batch=16, conf=0.001, iou=0.6,
                  device="mps", project=str(OUT), name=name, exist_ok=True, plots=name == "val_all")
    names = m.names
    per_class = {}
    for i, c in enumerate(m.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        per_class[names[int(c)]] = {"precision": round(float(p), 4), "recall": round(float(r), 4),
                                    "mAP50": round(float(ap50), 4), "mAP50_95": round(float(ap), 4)}
    overall = {"precision": round(float(m.box.mp), 4), "recall": round(float(m.box.mr), 4),
               "mAP50": round(float(m.box.map50), 4), "mAP50_95": round(float(m.box.map), 4)}
    return overall, per_class


overall, per_class = run(DATA / "data.yaml", "val_all")

# per-country breakdown: build a temporary yaml pointing at a file list per country prefix
per_country = {}
val_imgs = sorted((DATA / "images/val").glob("*.jpg"))
tmp = pathlib.Path(tempfile.mkdtemp())
for country in ["China_Drone", "China_MotorBike", "Czech", "India", "Japan", "United_States"]:
    files = [str(p.resolve()) for p in val_imgs if p.name.startswith(country + "_")]
    if not files:
        continue
    lst = tmp / f"{country}.txt"
    lst.write_text("\n".join(files))
    y = tmp / f"{country}.yaml"
    y.write_text(f"path: {DATA.resolve()}\ntrain: {lst}\nval: {lst}\nnames:\n" +
                 "".join(f"  {k}: {v}\n" for k, v in model.names.items()))
    o, _ = run(y, f"val_{country}")
    per_country[country] = {"images": len(files), **o}
shutil.rmtree(tmp)

stats = json.loads((DATA / "stats.json").read_text())
metrics = {"model": "YOLO11n", "weights": pathlib.Path(W).name, "dataset": "RDD2022 (6 countries, Norway excluded)",
           "val_images": stats["images"]["val"], "train_images": stats["images"]["train"],
           "imgsz": 640, "overall": overall, "per_class": per_class, "per_country": per_country}
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

md = ["| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |", "|---|---|---|---|---|"]
for k, v in per_class.items():
    md.append(f"| {k} | {v['precision']:.3f} | {v['recall']:.3f} | {v['mAP50']:.3f} | {v['mAP50_95']:.3f} |")
md.append(f"| **all** | **{overall['precision']:.3f}** | **{overall['recall']:.3f}** | **{overall['mAP50']:.3f}** | **{overall['mAP50_95']:.3f}** |")
md += ["", "| Country | Val images | mAP@0.5 | mAP@0.5:0.95 |", "|---|---|---|---|"]
for k, v in per_country.items():
    md.append(f"| {k} | {v['images']} | {v['mAP50']:.3f} | {v['mAP50_95']:.3f} |")
(OUT / "RESULTS.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
