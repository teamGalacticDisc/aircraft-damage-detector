"""Convert mixed polygon rows in YOLO label files to bounding boxes.

Box rows (5 numbers) are kept as is. Polygon rows (more than 5 numbers) are
converted to their bounding box, clamped to 0..1. A converted polygon box is
dropped if it overlaps an existing box row in the same file with IoU > 0.5.
Output goes to data_fixed with images copied (not linked).
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "data"
DST = ROOT / "data_fixed"
SPLITS = ["train", "valid", "test"]
IOU_THRESHOLD = 0.5


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def clamp(v):
    return min(1.0, max(0.0, v))


def main():
    files_changed = polys_converted = polys_dropped = 0
    for split in SPLITS:
        src_img, src_lbl = SRC / split / "images", SRC / split / "labels"
        dst_img, dst_lbl = DST / split / "images", DST / split / "labels"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        for img in src_img.iterdir():
            if img.is_file():
                shutil.copy2(img, dst_img / img.name)
        for lbl in src_lbl.glob("*.txt"):
            boxes, polys = [], []
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if not parts:
                    continue
                cls = int(float(parts[0]))
                nums = [float(p) for p in parts[1:]]
                if len(nums) == 4:
                    boxes.append((cls, nums))
                else:
                    xs = [clamp(v) for v in nums[0::2]]
                    ys = [clamp(v) for v in nums[1::2]]
                    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
                    polys.append((cls, [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]))
            lines = []
            for line in lbl.read_text().splitlines():
                if len(line.split()) == 5:
                    lines.append(line.strip())
            if polys:
                files_changed += 1
                for cls, pb in polys:
                    if any(iou(pb, bb) > IOU_THRESHOLD for _, bb in boxes):
                        polys_dropped += 1
                        continue
                    polys_converted += 1
                    lines.append(f"{cls} " + " ".join(f"{v:.6f}" for v in pb))
            (dst_lbl / lbl.name).write_text("\n".join(lines) + ("\n" if lines else ""))
    yaml = (
        f"path: {DST.as_posix()}\n"
        "train: train/images\nval: valid/images\ntest: test/images\n\n"
        "nc: 2\nnames: ['crack', 'dent']\n"
    )
    (DST / "dataset.yaml").write_text(yaml)
    print(f"files with polygon rows: {files_changed}")
    print(f"polygon rows converted to boxes: {polys_converted}")
    print(f"polygon boxes dropped as duplicates (IoU > {IOU_THRESHOLD}): {polys_dropped}")


if __name__ == "__main__":
    sys.exit(main())
