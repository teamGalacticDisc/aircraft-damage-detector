# Makes a new split where every copy of the same original photo stays in one group.
#
# The Roboflow download spread brighter, darker and flipped copies of each photo
# across train, valid and test. That lets the model study a copy of a quiz photo.
# This script puts each original photo and all of its copies into only one group.
#
# Run fix_labels.py first. This reads data_fixed and writes data_clean.

import csv
import os
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).parent
SRC = ROOT / "data_fixed"
DST = ROOT / "data_clean"
SHARES = {"train": 0.70, "valid": 0.15, "test": 0.15}
SEED = 0
LOOKALIKE = 0.97  # two photos this similar or more count as the same original


def name_group(filename):
    """Copies share a number before the first underscore, or the name before .rf."""
    stem = filename.split(".rf.")[0]
    m = re.match(r"(\d+)_", stem)
    return "N" + m.group(1) if m else stem


def photo_vector(path, flip=False):
    """A tiny 32 by 32 gray copy of the photo, adjusted so brightness and contrast do not matter."""
    img = Image.open(path).convert("L")
    if flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    a = np.asarray(img.resize((32, 32)), dtype=np.float32).flatten()
    a -= a.mean()
    return a / (np.linalg.norm(a) + 1e-6)


def main():
    images = []
    for split in ["train", "valid", "test"]:
        for f in sorted(os.listdir(SRC / split / "images")):
            images.append((split, f))

    # 1. Group by file name
    parent = {}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        parent[find(a)] = find(b)

    groups = [name_group(f) for _, f in images]
    for g in groups:
        parent.setdefault(g, g)

    # 2. Also join groups whose photos look almost the same, in case a copy was renamed.
    #    A flipped copy is checked too. I looked at sample pairs by eye: 0.97 and up were
    #    always the same photo, and below that most were different photos that were just
    #    smooth gray metal.
    vecs = np.array([photo_vector(SRC / s / "images" / f) for s, f in images])
    flips = np.array([photo_vector(SRC / s / "images" / f, flip=True) for s, f in images])
    sim = np.maximum(vecs @ vecs.T, vecs @ flips.T)
    joined = 0
    for i, j in zip(*np.nonzero(np.triu(sim, 1) >= LOOKALIKE)):
        if find(groups[i]) != find(groups[j]):
            union(groups[i], groups[j])
            joined += 1
    final_group = [find(g) for g in groups]

    members = defaultdict(list)
    for (s, f), g in zip(images, final_group):
        members[g].append((s, f))

    # 3. Count photos, cracks and dents in every group
    def tally(g):
        t = np.zeros(3)
        for s, f in members[g]:
            t[0] += 1
            for line in open(SRC / s / "labels" / (Path(f).stem + ".txt")):
                if line.strip():
                    t[1 + int(line.split()[0])] += 1
        return t
    sizes = {g: tally(g) for g in members}
    total = sum(sizes.values())

    # 4. Try many random ways to deal the groups out, like dealing cards, and keep the one
    #    where every split gets close to its share of photos, cracks and dents.
    def deal(seed):
        rng = random.Random(seed)
        order = sorted(members)
        rng.shuffle(order)
        counts = {k: np.zeros(3) for k in SHARES}
        assign = {}
        for g in order:
            target = min(SHARES, key=lambda k: (counts[k] / (SHARES[k] * total)).max())
            assign[g] = target
            counts[target] += sizes[g]
        miss = max(abs(counts[k] / total - SHARES[k]).max() for k in SHARES)
        return miss, assign, counts
    miss, assign, counts = min((deal(SEED + i) for i in range(500)), key=lambda r: r[0])
    counts = {k: int(v[0]) for k, v in counts.items()}

    # 5. Write data_clean and a manifest that lists every photo, its group and its split
    if DST.exists():
        shutil.rmtree(DST)
    for k in SHARES:
        (DST / k / "images").mkdir(parents=True)
        (DST / k / "labels").mkdir(parents=True)
    with open(ROOT / "results" / "clean_split_manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "original_split", "group", "new_split"])
        for (s, f), g in zip(images, final_group):
            k = assign[g]
            w.writerow([f, s, g, k])
            shutil.copy2(SRC / s / "images" / f, DST / k / "images" / f)
            lab = Path(f).with_suffix(".txt").name
            shutil.copy2(SRC / s / "labels" / lab, DST / k / "labels" / lab)

    (DST / "dataset.yaml").write_text(
        f"path: {DST.resolve().as_posix()}\n"
        "train: train/images\nval: valid/images\ntest: test/images\n"
        "nc: 2\nnames: ['crack', 'dent']\n"
    )

    print(f"{len(images)} photos, {len(members)} original photo groups, "
          f"{joined} groups joined because they looked alike")
    for k in SHARES:
        n_groups = sum(1 for g in assign if assign[g] == k)
        print(f"{k}: {counts[k]} photos from {n_groups} original photos")


def from_manifest():
    """Rebuild data_clean from the saved manifest, so everyone gets the exact same split."""
    if DST.exists():
        shutil.rmtree(DST)
    for k in SHARES:
        (DST / k / "images").mkdir(parents=True)
        (DST / k / "labels").mkdir(parents=True)
    rows = list(csv.DictReader(open(ROOT / "results" / "clean_split_manifest.csv")))
    for r in rows:
        s, f, k = r["original_split"], r["image"], r["new_split"]
        shutil.copy2(SRC / s / "images" / f, DST / k / "images" / f)
        lab = Path(f).stem + ".txt"
        shutil.copy2(SRC / s / "labels" / lab, DST / k / "labels" / lab)
    (DST / "dataset.yaml").write_text(
        f"path: {DST.resolve().as_posix()}\n"
        "train: train/images\nval: valid/images\ntest: test/images\n"
        "nc: 2\nnames: ['crack', 'dent']\n"
    )
    print(f"rebuilt data_clean from the manifest: {len(rows)} photos")


if __name__ == "__main__":
    import sys
    if "--use-manifest" in sys.argv:
        from_manifest()
    else:
        main()
