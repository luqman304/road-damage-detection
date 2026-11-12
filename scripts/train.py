"""Train YOLO11 on RDD2022 (YOLO format).

Ultralytics forces workers=0 on Apple MPS, which makes augmentation the bottleneck
on an M-series Mac. This trainer restores parallel dataloader workers.

usage: python scripts/train.py MODEL EPOCHS NAME [FRACTION] [BATCH]
"""
import sys
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

WORKERS = 8


class MPSWorkersTrainer(DetectionTrainer):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.args.workers = WORKERS


if __name__ == "__main__":
    model, epochs, name = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    fraction = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    batch = int(sys.argv[5]) if len(sys.argv) > 5 else 16
    YOLO(model).train(
        trainer=MPSWorkersTrainer,
        data="data/rdd2022_yolo/data.yaml",
        epochs=epochs, imgsz=640, batch=batch, device="mps",
        fraction=fraction, project="runs", name=name, exist_ok=True,
        cos_lr=True, close_mosaic=5, patience=0,
        fliplr=0.5, mosaic=1.0, mixup=0.0,
        plots=True, val=fraction == 1.0,
    )
