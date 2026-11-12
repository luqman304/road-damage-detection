"""Convert RDD2022 Pascal-VOC annotations to a YOLO dataset.

Keeps the 4 official CRDDC classes (D00, D10, D20, D40), drops other labels
(D43, D44, D50, Repair, ...). Only the RDD2022 *train* folders have labels, so we
make a seeded 85/15 train/val split per country. Images are hard-linked, not copied.
"""
import os, sys, random, json, pathlib, xml.etree.ElementTree as ET
from collections import Counter

RAW = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
OUT = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "data/rdd2022_yolo")
CLASSES = ["D00", "D10", "D20", "D40"]
NAMES = {
    0: "longitudinal_crack",
    1: "transverse_crack",
    2: "alligator_crack",
    3: "pothole",
}
VAL_FRAC = 0.15
random.seed(0)


def convert(xml_path):
    root = ET.parse(xml_path).getroot()
    w = float(root.find("size/width").text)
    h = float(root.find("size/height").text)
    lines = []
    for obj in root.iter("object"):
        name = obj.find("name").text.strip()
        if name not in CLASSES:
            continue
        b = obj.find("bndbox")
        x1, y1, x2, y2 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        lines.append(f"{CLASSES.index(name)} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
    return lines


stats = {"train": Counter(), "val": Counter()}
images = {"train": 0, "val": 0}
for country_dir in sorted(p for p in RAW.iterdir() if p.is_dir() and not p.name.startswith("_")):
    xmls = sorted((country_dir / "train/annotations/xmls").glob("*.xml"))
    random.shuffle(xmls)
    n_val = int(len(xmls) * VAL_FRAC)
    for i, xml in enumerate(xmls):
        split = "val" if i < n_val else "train"
        img = country_dir / "train/images" / (xml.stem + ".jpg")
        if not img.exists():
            continue
        lines = convert(xml)  # images with no damage are kept as background negatives
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        dst = OUT / "images" / split / img.name
        if not dst.exists():
            os.link(img, dst)
        (OUT / "labels" / split / (xml.stem + ".txt")).write_text("\n".join(lines))
        images[split] += 1
        for l in lines:
            stats[split][NAMES[int(l.split()[0])]] += 1
    print(f"{country_dir.name}: {len(xmls)} images", flush=True)

yaml = f"path: {OUT.resolve()}\ntrain: images/train\nval: images/val\nnames:\n" + "".join(
    f"  {k}: {v}\n" for k, v in NAMES.items())
(OUT / "data.yaml").write_text(yaml)
summary = {"images": images, "instances": {k: dict(v) for k, v in stats.items()}}
(OUT / "stats.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
