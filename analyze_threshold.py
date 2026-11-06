# Checks what happens at different confidence cutoffs, and puts error bars on the numbers.
#
# For each cutoff it reports:
#   crack and dent recall   (out of all the real damage boxes, how many did it find)
#   false alarms per photo  (boxes drawn where nothing was marked)
#   photo recall            (out of the photos with real damage, how many got at least one correct box)
# Error bars come from resampling whole original photo groups, so copies of the
# same photo are never treated as separate proof.
#
# Usage: python analyze_threshold.py model.pt valid   (or test, only once at the very end)

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from ultralytics import YOLO

HERE = Path(__file__).parent
DATA = HERE / "data_clean"
CUTOFFS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
BOOTSTRAPS = 1000


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0


def load_truth(label_path):
    boxes = []
    for line in open(label_path):
        v = line.split()
        if not v:
            continue
        c, x, y, w, h = int(v[0]), *map(float, v[1:5])
        boxes.append((c, (x - w / 2, y - h / 2, x + w / 2, y + h / 2)))
    return boxes


def score_photo(truth, preds, cutoff):
    """Match boxes one to one, most confident first, same class, overlap 0.5 or more."""
    p = sorted([q for q in preds if q[2] >= cutoff], key=lambda q: -q[2])
    used = set()
    hits = [0, 0]
    false_alarms = 0
    for c, box, conf in p:
        best, best_i = 0.5, None
        for i, (tc, tb) in enumerate(truth):
            if i in used or tc != c:
                continue
            o = iou(box, tb)
            if o >= best:
                best, best_i = o, i
        if best_i is None:
            false_alarms += 1
        else:
            used.add(best_i)
            hits[c] += 1
    totals = [sum(1 for tc, _ in truth if tc == k) for k in (0, 1)]
    return hits, totals, false_alarms, len(used) > 0


def summarize(rows):
    h = np.sum([r[0] for r in rows], axis=0)
    t = np.sum([r[1] for r in rows], axis=0)
    fa = sum(r[2] for r in rows)
    damaged = [r for r in rows if sum(r[1]) > 0]
    return {
        "crack_recall": h[0] / max(t[0], 1),
        "dent_recall": h[1] / max(t[1], 1),
        "false_alarms_per_photo": fa / len(rows),
        "photo_recall": sum(r[3] for r in damaged) / max(len(damaged), 1),
    }


def main(weights, split):
    manifest = {r["image"]: r["group"] for r in csv.DictReader(open(HERE / "results" / "clean_split_manifest.csv"))}
    img_dir = DATA / split / "images"
    model = YOLO(weights)
    per_photo = {}
    for res in model.predict(source=str(img_dir), conf=0.01, imgsz=640, stream=True, verbose=False, device="cpu"):
        name = Path(res.path).name
        preds = [(int(c), tuple(b), float(s)) for c, b, s in
                 zip(res.boxes.cls.tolist(), res.boxes.xyxyn.tolist(), res.boxes.conf.tolist())]
        truth = load_truth(DATA / split / "labels" / (Path(name).stem + ".txt"))
        per_photo[name] = (truth, preds)

    groups = defaultdict(list)
    for name in per_photo:
        groups[manifest[name]].append(name)
    group_ids = sorted(groups)
    rng = random.Random(0)

    report = {"weights": str(weights), "split": split, "photos": len(per_photo),
              "original_photos": len(group_ids), "cutoffs": {}}
    for cut in CUTOFFS:
        scored = {n: score_photo(t, p, cut) for n, (t, p) in per_photo.items()}
        point = summarize(list(scored.values()))
        boots = defaultdict(list)
        for _ in range(BOOTSTRAPS):
            pick = [rng.choice(group_ids) for _ in group_ids]
            s = summarize([scored[n] for g in pick for n in groups[g]])
            for k, v in s.items():
                boots[k].append(v)
        report["cutoffs"][str(cut)] = {
            k: {"value": round(v, 3),
                "low": round(float(np.percentile(boots[k], 2.5)), 3),
                "high": round(float(np.percentile(boots[k], 97.5)), 3)}
            for k, v in point.items()
        }
        c = report["cutoffs"][str(cut)]
        print(f"cutoff {cut:.2f}: cracks {c['crack_recall']['value']:.2f} "
              f"[{c['crack_recall']['low']:.2f} to {c['crack_recall']['high']:.2f}], "
              f"dents {c['dent_recall']['value']:.2f}, "
              f"false alarms per photo {c['false_alarms_per_photo']['value']:.2f}, "
              f"photos flagged {c['photo_recall']['value']:.2f}")
    return report


if __name__ == "__main__":
    w, sp = sys.argv[1], sys.argv[2]
    out = main(w, sp)
    Path(HERE / "results" / "clean").mkdir(parents=True, exist_ok=True)
    (HERE / "results" / "clean" / f"threshold_{Path(w).stem}_{sp}.json").write_text(json.dumps(out, indent=2))
