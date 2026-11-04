# Grades the trained model on pictures it never saw while it was learning.

from pathlib import Path
from ultralytics import YOLO

here = Path(__file__).parent
model = YOLO(here / "model" / "best.pt")

for split in ["val", "test"]:
    m = model.val(data=str(here / "data" / "dataset.yaml"), split=split, imgsz=640)
    print(f"{split}: mAP50 {m.box.map50:.3f}, precision {m.box.mp:.3f}, recall {m.box.mr:.3f}")
