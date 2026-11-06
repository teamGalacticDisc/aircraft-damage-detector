# Opens the locked test photos one time, for the setup that won on validation.
#
# The winner was picked using only the validation photos (see run_experiments.py).
# This grades its 3 seeds on the 452 test photos from 28 original photos the model
# never saw in any form, then runs the cutoff check with error bars.
#
# Usage: python final_test.py

import json
from pathlib import Path

from ultralytics import YOLO

import analyze_threshold

HERE = Path(__file__).parent
DATA = HERE / "data_clean" / "dataset.yaml"
WINNER = "B_v8s"
SEEDS = [0, 1, 2]

out = {}
for s in SEEDS:
    w = HERE / "clean_results" / f"{WINNER}_s{s}_best.pt"
    m = YOLO(w).val(data=str(DATA), split="test", imgsz=640, device="cpu", plots=False)
    out[f"seed{s}"] = {
        "test_mAP50": m.box.map50, "test_mAP50_95": m.box.map,
        "precision": m.box.mp, "recall": m.box.mr,
        "crack_mAP50": float(m.box.ap50[0]), "dent_mAP50": float(m.box.ap50[1]),
        "cutoffs": analyze_threshold.main(w, "test")["cutoffs"],
    }
(HERE / "results" / "clean").mkdir(parents=True, exist_ok=True)
(HERE / "results" / "clean" / "final_test.json").write_text(json.dumps(out, indent=2))
print(json.dumps({k: round(v["test_mAP50"], 3) for k, v in out.items()}))
