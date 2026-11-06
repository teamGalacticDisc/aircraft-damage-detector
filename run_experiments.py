# Tests the ideas from the three AI reviews on the clean split.
#
# Every setup is trained 3 times with different random starts (seeds), because one run
# can get lucky. Only the validation photos are used here. The test photos stay locked
# away until one setup has been picked, and then final_test.py opens them one time.
#
# Usage (one graphics card per window):
#   python run_experiments.py --gpu 0 --workers 3 --runs A_base_s0,A_base_s1
#   python run_experiments.py --gpu 1 --runs B_v8s_s0

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).parent
DATA = HERE / "data_clean" / "dataset.yaml"
OUT = HERE / "clean_results"

# What each setup changes, and which review asked for it
SETUPS = {
    "A_base": ("yolov8n.pt", {}),                                         # all three: same final settings, clean split
    "B_v8s": ("yolov8s.pt", {}),                                          # ChatGPT and Claude: recheck small vs bigger model
    "C_freeze": ("yolov8n.pt", {"freeze": 10}),                           # Claude: lock the first 10 layers
    "D_moreaug": ("yolov8n.pt", {"degrees": 10, "mixup": 0.15, "translate": 0.2}),  # Claude: stronger augmentation
    "E_clspw": ("yolov8n.pt", {"cls_pw": 1.0}),                           # my docs research: extra weight for the weaker class
}


def run(name, gpu, workers):
    setup, seed = name.rsplit("_s", 1)
    weights, extra = SETUPS[setup]
    out = OUT / f"{name}.json"
    if out.exists():
        print(f"{name} already done, skipping")
        return
    model = YOLO(weights)
    model.train(
        data=str(DATA), epochs=60, patience=15, imgsz=640, batch=16,
        seed=int(seed), deterministic=True, device=gpu, workers=workers,
        project=str(HERE / "runs_clean"), name=name, exist_ok=True, **extra,
    )
    best = HERE / "runs_clean" / name / "weights" / "best.pt"
    m = YOLO(best).val(data=str(DATA), split="val", imgsz=640, device=gpu, workers=workers,
                      project=str(HERE / "runs_clean"), name=name + "_val", exist_ok=True)
    rows = (HERE / "runs_clean" / name / "results.csv").read_text().strip().splitlines()
    header = [h.strip() for h in rows[0].split(",")]
    col = header.index("metrics/mAP50(B)")
    per_epoch = [float(r.split(",")[col]) for r in rows[1:]]
    result = {
        "name": name, "setup": setup, "seed": int(seed), "weights": weights, "extra": extra,
        "epochs_run": len(per_epoch), "best_epoch": per_epoch.index(max(per_epoch)) + 1,
        "val_mAP50": m.box.map50, "val_mAP50_95": m.box.map,
        "val_precision": m.box.mp, "val_recall": m.box.mr,
        "crack_mAP50": float(m.box.ap50[0]), "dent_mAP50": float(m.box.ap50[1]),
    }
    OUT.mkdir(exist_ok=True)
    (OUT / f"{name}_best.pt").write_bytes(best.read_bytes())
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":  # needed on Windows
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--runs", required=True)
    # Fewer loader workers per card. On Windows each worker loads its own copy of PyTorch,
    # and 8 per card on two cards ran the PC out of memory.
    p.add_argument("--workers", type=int, default=3)
    a = p.parse_args()
    for name in a.runs.split(","):
        try:
            run(name, a.gpu, a.workers)
        except Exception as e:  # keep going so one bad setting does not stop the whole queue
            OUT.mkdir(exist_ok=True)
            (OUT / f"{name}_error.txt").write_text(repr(e))
            print(f"{name} failed: {e!r}")
