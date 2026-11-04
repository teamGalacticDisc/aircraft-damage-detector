# Teaches a YOLOv8 model to find cracks and dents on aircraft skin.
#
# Before you run this:
#   1. Download the Roboflow dataset aircraft-damage-detection-1j9qk, version 5,
#      in YOLOv8 format.
#   2. Unzip it into a folder called data right next to this file.
#      You should see data/train, data/valid and data/test.
#   3. pip install -r requirements.txt

from pathlib import Path
from ultralytics import YOLO

DATA_DIR = Path(__file__).parent / "data"

# Roboflow's own data.yaml uses paths that break if you run from another folder,
# so we write a small one that always points to the right place.
yaml_file = DATA_DIR / "dataset.yaml"
yaml_file.write_text(
    f"path: {DATA_DIR.resolve()}\n"
    "train: train/images\n"
    "val: valid/images\n"
    "test: test/images\n"
    "nc: 2\n"
    "names: ['crack', 'dent']\n"
)

# Start from the small pretrained YOLOv8 model. It already knows what edges,
# corners and shapes look like, so it only has to learn what a crack and a dent are.
model = YOLO("yolov8n.pt")

model.train(
    data=str(yaml_file),
    epochs=20,      # look at every training picture 20 times
    imgsz=640,      # the pictures are already 640 by 640
    batch=16,
    seed=0,         # same seed means you get the same result if you run it again
    name="v5_yolov8n",
)
