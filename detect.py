# Runs the trained model on your own picture and saves a copy with boxes drawn on it.
# Example:  python detect.py my_wing_photo.jpg

import sys
from pathlib import Path
from ultralytics import YOLO

model = YOLO(Path(__file__).parent / "model" / "best.pt")
results = model.predict(sys.argv[1], conf=0.25, save=True)
for box in results[0].boxes:
    name = model.names[int(box.cls)]
    print(f"Found a {name}, confidence {float(box.conf):.2f}")
print("Picture with boxes saved in runs/detect/predict")
