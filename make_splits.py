#!/usr/bin/env python3
"""Regenerate the WET-Net train/test split index files (Section IV-A).

For each dataset the pooled, publicly released labelled images are divided ONCE into a
training and a test partition in a 7:3 ratio with a fixed seed of 42:

    isic2018   seeded random split (diagnostic labels not consistently available)
    isic2017   stratified by diagnostic category
    ph2        stratified by diagnostic category
    ham10000   stratified by diagnostic category, split AT LESION LEVEL (lesion_id),
               so that no lesion contributes images to both partitions

Each partition is written as a plain index file -- one image identifier (file stem) per
line -- into this folder, in the exact format the data loader reads
(``wetnet/data.py::SkinLesionDataset``).

Examples
--------
    python splits/make_splits.py isic2018 --image-dir data/isic2018/all/images
    python splits/make_splits.py isic2017 --image-dir data/isic2017/all/images \
                                           --labels-csv data/isic2017/labels.csv
    python splits/make_splits.py ph2      --labels-csv data/ph2/labels.csv
    python splits/make_splits.py ham10000 --metadata-csv data/ham10000/HAM10000_metadata.csv

Use the generator only to *regenerate* the protocol; if the original .txt files used for the
reported numbers still exist, commit those instead and check them with verify_splits.py.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

SEED = 42
TEST_SIZE = 0.30
HERE = Path(__file__).resolve().parent

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# (train, test) counts stated in the manuscript, for a sanity warning only.
EXPECTED = {
    "isic2018": (1886, 808),
    "isic2017": (1500, 650),
    "ham10000": (7010, 3005),
    "ph2": (140, 60),
}


# ----------------------------------------------------------------------------- helpers
def _require_sklearn():
    try:
        import sklearn  # noqa: F401
    except ImportError:
        sys.exit("scikit-learn is required: pip install scikit-learn")


def list_stems(image_dir: Path) -> list[str]:
    if not image_dir.is_dir():
        sys.exit(f"--image-dir not found: {image_dir}")
    stems = sorted(p.stem for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    if not stems:
        sys.exit(f"no images with extensions {sorted(IMG_EXT)} in {image_dir}")
    return stems


def read_labels_csv(path: Path, id_col: str, label_col: str) -> dict[str, str]:
    if not path.is_file():
        sys.exit(f"--labels-csv not found: {path}")
    out: dict[str, str] = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        _check_cols(reader.fieldnames, (id_col, label_col), path)
        for row in reader:
            stem = Path(str(row[id_col]).strip()).stem
            out[stem] = str(row[label_col]).strip()
    if not out:
        sys.exit(f"no rows read from {path}")
    return out


def read_ham_metadata(path: Path, id_col: str, group_col: str, label_col: str):
    if not path.is_file():
        sys.exit(f"--metadata-csv not found: {path}")
    ids, groups, labels = [], [], []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        _check_cols(reader.fieldnames, (id_col, group_col, label_col), path)
        for row in reader:
            ids.append(Path(str(row[id_col]).strip()).stem)
            groups.append(str(row[group_col]).strip())
            labels.append(str(row[label_col]).strip())
    if not ids:
        sys.exit(f"no rows read from {path}")
    return ids, groups, labels


def _check_cols(fieldnames, needed, path):
    missing = [c for c in needed if not fieldnames or c not in fieldnames]
    if missing:
        sys.exit(f"{path} is missing column(s) {missing}; found {list(fieldnames or [])}. "
                 f"Use --id-col / --label-col / --group-col to match your CSV.")


def write_index(stems, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(sorted(stems)) + "\n")


def _report(name: str, train, test) -> None:
    n = len(train) + len(test)
    frac = len(test) / n if n else 0.0
    print(f"  {name}: {len(train)} train + {len(test)} test = {n}  (test fraction {frac:.3f})")
    exp = EXPECTED.get(name)
    if exp and (len(train), len(test)) != exp:
        print(f"  NOTE: manuscript states {exp[0]} train / {exp[1]} test for {name}. "
              f"A small difference is the expected effect of stratification rounding on a "
              f"different pool; a large one means the pool differs from the paper's.")
    # overlap guard (should be impossible, but assert it)
    dup = set(train) & set(test)
    if dup:
        sys.exit(f"INTERNAL ERROR: {len(dup)} ids in both partitions, e.g. {sorted(dup)[:3]}")


# ----------------------------------------------------------------------------- splitters
def split_random(stems):
    from sklearn.model_selection import train_test_split
    tr, te = train_test_split(stems, test_size=TEST_SIZE, random_state=SEED, shuffle=True)
    return tr, te


def split_stratified(stems, label_of):
    from sklearn.model_selection import train_test_split
    stems = [s for s in stems if s in label_of]
    if not stems:
        sys.exit("none of the image stems were found in the labels CSV; check --id-col / paths")
    y = [label_of[s] for s in stems]
    # Drop singleton classes from stratification (cannot appear in both partitions).
    from collections import Counter
    counts = Counter(y)
    strat = y if all(c >= 2 for c in counts.values()) else None
    if strat is None:
        print("  WARNING: a diagnostic category has <2 samples; falling back to random split "
              "for those. Stratification needs at least 2 per class.")
    tr, te = train_test_split(stems, test_size=TEST_SIZE, random_state=SEED,
                              shuffle=True, stratify=strat)
    return tr, te


def split_grouped_stratified(ids, groups, labels):
    """Lesion-level split: no lesion_id in both partitions, class balance kept where possible."""
    import numpy as np
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        # n_splits=3 -> ~1/3 test; pick the fold whose test fraction is closest to 0.30.
        skf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
        best = None
        for tr_idx, te_idx in skf.split(ids, labels, groups):
            frac = len(te_idx) / len(ids)
            if best is None or abs(frac - TEST_SIZE) < abs(best[0] - TEST_SIZE):
                best = (frac, tr_idx, te_idx)
        _, tr_idx, te_idx = best
    except ImportError:
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
        tr_idx, te_idx = next(gss.split(ids, labels, groups))
        print("  NOTE: StratifiedGroupKFold unavailable; used GroupShuffleSplit "
              "(lesion-disjoint, not class-stratified). Upgrade scikit-learn for stratification.")
    ids = np.asarray(ids)
    return list(ids[tr_idx]), list(ids[te_idx])


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["isic2018", "isic2017", "ham10000", "ph2"])
    ap.add_argument("--image-dir", type=Path,
                    help="folder of images; required for isic2018, optional elsewhere "
                         "(used to intersect with the labels CSV)")
    ap.add_argument("--labels-csv", type=Path,
                    help="CSV with image id and diagnostic label (isic2017, ph2)")
    ap.add_argument("--metadata-csv", type=Path,
                    help="HAM10000 metadata CSV with image_id, lesion_id, dx")
    ap.add_argument("--id-col", default=None, help="id column (default: image_id for ham10000, else 'image')")
    ap.add_argument("--label-col", default=None, help="label column (default: 'dx' for ham10000, else 'label')")
    ap.add_argument("--group-col", default="lesion_id", help="HAM10000 group column (default: lesion_id)")
    ap.add_argument("--out-dir", type=Path, default=HERE, help="output folder (default: this folder)")
    a = ap.parse_args()

    _require_sklearn()
    ds = a.dataset
    print(f"Generating {ds} split  (seed={SEED}, test_size={TEST_SIZE})")

    if ds == "isic2018":
        if not a.image_dir:
            sys.exit("isic2018 needs --image-dir (the pooled image folder)")
        tr, te = split_random(list_stems(a.image_dir))

    elif ds in ("isic2017", "ph2"):
        if not a.labels_csv:
            sys.exit(f"{ds} is stratified; pass --labels-csv with an image/label column")
        id_col = a.id_col or "image"
        label_col = a.label_col or "label"
        label_of = read_labels_csv(a.labels_csv, id_col, label_col)
        stems = list_stems(a.image_dir) if a.image_dir else sorted(label_of)
        tr, te = split_stratified(stems, label_of)

    else:  # ham10000
        if not a.metadata_csv:
            sys.exit("ham10000 needs --metadata-csv (image_id, lesion_id, dx)")
        id_col = a.id_col or "image_id"
        label_col = a.label_col or "dx"
        ids, groups, labels = read_ham_metadata(a.metadata_csv, id_col, a.group_col, label_col)
        if a.image_dir:
            present = set(list_stems(a.image_dir))
            keep = [i for i, s in enumerate(ids) if s in present]
            ids = [ids[i] for i in keep]; groups = [groups[i] for i in keep]; labels = [labels[i] for i in keep]
        tr, te = split_grouped_stratified(ids, groups, labels)

    _report(ds, tr, te)
    train_path = a.out_dir / f"{ds}_train.txt"
    test_path = a.out_dir / f"{ds}_test.txt"
    write_index(tr, train_path)
    write_index(te, test_path)
    print(f"  wrote {train_path}")
    print(f"  wrote {test_path}")


if __name__ == "__main__":
    main()
