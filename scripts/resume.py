"""Resume an interrupted run from last.pt, keeping parallel dataloader workers on MPS."""
import sys
sys.path.insert(0, "scripts")
from train import MPSWorkersTrainer
from ultralytics import YOLO

if __name__ == "__main__":
    YOLO(sys.argv[1]).train(resume=True, trainer=MPSWorkersTrainer)
