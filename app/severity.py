"""Severity scoring for road-damage detections.

Per detection (0-100):
    severity = 100 * class_weight * extent * (0.5 + 0.5 * confidence)

  class_weight  structural risk of the damage type
                pothole 1.0 > alligator crack 0.85 > transverse 0.55 > longitudinal 0.45
                (potholes/alligator cracking indicate base failure; linear cracks are early-stage)
  extent        how much of the road the defect covers, from the box area relative
                to the image, log-scaled so small defects are not ignored and huge
                boxes saturate:  extent = clip(0.35 + 0.65 * log10(1 + 99*a)/2, 0, 1)
                where a = box_area / image_area (a=1% -> ~0.68, a=10% -> ~0.99)
  confidence    down-weights uncertain detections (factor in [0.5, 1])

Levels: low < 35 <= medium < 65 <= high.

Image-level Road Condition Index (RCI, 0-100, higher = better road), inspired by PCI
deduct-value logic: deductions from each detection are combined with diminishing
returns so many minor cracks do not outweigh one big pothole the wrong way.
    RCI = 100 * prod(1 - severity_i / 150)
"""
from __future__ import annotations
import math

CLASS_WEIGHTS = {
    "pothole": 1.0,
    "alligator_crack": 0.85,
    "transverse_crack": 0.55,
    "longitudinal_crack": 0.45,
}
REPAIR_ACTION = {
    "high": "Immediate repair recommended",
    "medium": "Schedule repair",
    "low": "Monitor",
}


def detection_severity(cls_name: str, conf: float, box_xyxy, img_w: int, img_h: int) -> dict:
    x1, y1, x2, y2 = box_xyxy
    area_frac = max(0.0, (x2 - x1) * (y2 - y1)) / float(img_w * img_h)
    extent = min(1.0, max(0.0, 0.35 + 0.65 * math.log10(1 + 99 * area_frac) / 2))
    score = 100.0 * CLASS_WEIGHTS.get(cls_name, 0.5) * extent * (0.5 + 0.5 * conf)
    score = round(min(100.0, score), 1)
    level = "high" if score >= 65 else "medium" if score >= 35 else "low"
    return {"severity_score": score, "severity_level": level, "area_fraction": round(area_frac, 4)}


def road_condition(severities: list[float]) -> dict:
    rci = 100.0
    for s in severities:
        rci *= 1 - s / 150.0
    rci = round(rci, 1)
    if rci >= 85:
        rating = "good"
    elif rci >= 60:
        rating = "fair"
    elif rci >= 35:
        rating = "poor"
    else:
        rating = "critical"
    worst = max(severities, default=0)
    level = "high" if worst >= 65 else "medium" if worst >= 35 else "low" if severities else "none"
    return {
        "road_condition_index": rci,
        "condition_rating": rating,
        "max_severity_level": level,
        "recommended_action": REPAIR_ACTION.get(level, "No action needed"),
    }
