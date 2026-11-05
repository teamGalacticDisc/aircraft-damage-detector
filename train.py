# Teaches a YOLOv8 model to find cracks and dents on aircraft skin.
#
# Before you run this:
#   1. Download the Roboflow dataset aircraft-damage-detection-1j9qk, version 5,
#      in YOLOv8 format.
#   2. Unzip it into a folder called data right next to this file.
#      You should see data/train, data/valid and data/test.
#   3. pip install -r requirements.txt
#   4. python fix_labels.py   (makes data_fixed so no photos get thrown out)

from pathlib import Path
from ultralytics import YOLO

DATA = Path(__file__).parent / "data_fixed" / "dataset.yaml"

# Start from the small pretrained YOLOv8 model. It already knows what edges,
# corners and shapes look like, so it only has to learn what a crack and a dent are.
model = YOLO("yolov8n.pt")

model.train(
    data=str(DATA),
    epochs=60,      # look at every training picture up to 60 times
    patience=15,    # stop early if the quiz score does not improve for 15 epochs
    imgsz=640,      # the pictures are already 640 by 640
    batch=16,
    seed=0,         # same seed means you get close to the same result if you run it again
    device=0,       # first graphics card; use device="cpu" if you do not have one
    name="fixed_v8n_60",
)
