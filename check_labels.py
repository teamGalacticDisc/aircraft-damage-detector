# Checks every label file before training, so a bad file is caught right away
# instead of YOLO quietly skipping the photo. (Google Gemini's fix 4.)
#
# Usage: python check_labels.py data_clean
# It stops with an error if any file has a problem.

import sys
from pathlib import Path


def check(root):
    problems = []
    photos = 0
    for split in ["train", "valid", "test"]:
        img_dir = Path(root) / split / "images"
        lab_dir = Path(root) / split / "labels"
        for img in sorted(img_dir.iterdir()):
            photos += 1
            lab = lab_dir / (img.stem + ".txt")
            if not lab.exists():
                problems.append(f"{split}/{img.name}: no label file")
                continue
            kinds = set()
            for n, line in enumerate(lab.read_text().splitlines(), 1):
                v = line.split()
                if not v:
                    continue
                if v[0] not in ("0", "1"):
                    problems.append(f"{lab.name} line {n}: class {v[0]} is not crack (0) or dent (1)")
                nums = list(map(float, v[1:]))
                if any(x < 0 or x > 1 for x in nums):
                    problems.append(f"{lab.name} line {n}: a number is outside 0 to 1")
                kinds.add("box" if len(nums) == 4 else "outline")
            if len(kinds) > 1:
                problems.append(f"{lab.name}: mixes boxes and outlines, YOLO would skip this photo")
    return photos, problems


if __name__ == "__main__":
    photos, problems = check(sys.argv[1] if len(sys.argv) > 1 else "data_clean")
    for p in problems:
        print(p)
    if problems:
        sys.exit(f"{len(problems)} problems found in {photos} photos")
    print(f"all {photos} photos have clean labels")
