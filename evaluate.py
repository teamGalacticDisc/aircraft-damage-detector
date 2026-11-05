# Grades the trained model on the validation and test pictures.
# Note: these groups hold brightened or flipped copies of photos that are also in
# the training group, so the scores run higher than they would on brand new photos.

from pathlib import Path
from ultralytics import YOLO

here = Path(__file__).parent
model = YOLO(here / "model" / "best.pt")
data = str(here / "data_fixed" / "dataset.yaml")

for split in ["val", "test"]:
    m = model.val(data=data, split=split, imgsz=640)
    print(f"{split}: mAP50 {m.box.map50:.3f}, precision {m.box.mp:.3f}, recall {m.box.mr:.3f}")

# Scoreboard at the same 25 percent setting detect.py uses
model.val(data=data, split="val", imgsz=640, conf=0.25, plots=True, name="val_conf025")
