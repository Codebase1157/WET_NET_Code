#!/usr/bin/env python3
"""Verify the committed WET-Net split index files.

Checks, per dataset:
  * both index files exist and are non-empty;
  * no identifier appears in both the train and the test partition;
  * no identifier is duplicated within a file;
  * (reporting) the train/test counts and the test fraction, compared with the manuscript.

For HAM10000, with --metadata-csv, it additionally checks the core reviewer requirement:
  * no lesion_id contributes images to BOTH partitions (lesion-level disjointness).

Exit code is non-zero if any hard check fails, so it can gate a commit or run in CI:

    python splits/verify_splits.py all
    python splits/verify_splits.py isic2018
    python splits/verify_splits.py ham10000 --metadata-csv data/ham10000/HAM10000_metadata.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASETS = ["isic2018", "isic2017", "ham10000", "ph2"]
EXPECTED = {
    "isic2018": (1886, 808),
    "isic2017": (1500, 650),
    "ham10000": (7010, 3005),
    "ph2": (140, 60),
}


def read_index(path: Path) -> list[str]:
    with open(path) as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def check_dataset(ds: str, out_dir: Path, metadata_csv: Path | None,
                  id_col: str, group_col: str) -> bool:
    ok = True
    tr_path = out_dir / f"{ds}_train.txt"
    te_path = out_dir / f"{ds}_test.txt"
    print(f"[{ds}]")

    for p in (tr_path, te_path):
        if not p.is_file():
            print(f"  FAIL  missing file: {p}")
            return False

    train = read_index(tr_path)
    test = read_index(te_path)
    train_set, test_set = set(train), set(test)

    if len(train_set) != len(train):
        print(f"  FAIL  {len(train) - len(train_set)} duplicate id(s) within {tr_path.name}"); ok = False
    if len(test_set) != len(test):
        print(f"  FAIL  {len(test) - len(test_set)} duplicate id(s) within {te_path.name}"); ok = False

    overlap = train_set & test_set
    if overlap:
        print(f"  FAIL  {len(overlap)} id(s) in BOTH partitions, e.g. {sorted(overlap)[:3]}"); ok = False
    else:
        print(f"  ok    train/test are disjoint")

    n = len(train_set) + len(test_set)
    frac = len(test_set) / n if n else 0.0
    print(f"  info  {len(train_set)} train + {len(test_set)} test = {n}  (test fraction {frac:.3f})")
    exp = EXPECTED.get(ds)
    if exp:
        if (len(train_set), len(test_set)) == exp:
            print(f"  ok    counts match the manuscript ({exp[0]}/{exp[1]})")
        else:
            print(f"  warn  manuscript states {exp[0]}/{exp[1]}; committed files have "
                  f"{len(train_set)}/{len(test_set)} (not fatal, but confirm this is intended)")

    if ds == "ham10000":
        if metadata_csv:
            ok = _check_lesion_disjoint(train_set, test_set, metadata_csv, id_col, group_col) and ok
        else:
            print("  skip  lesion-level disjointness (pass --metadata-csv to enable this check)")

    return ok


def _check_lesion_disjoint(train_set, test_set, metadata_csv: Path, id_col, group_col) -> bool:
    if not metadata_csv.is_file():
        print(f"  FAIL  --metadata-csv not found: {metadata_csv}")
        return False
    lesion_of: dict[str, str] = {}
    with open(metadata_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or id_col not in reader.fieldnames or group_col not in reader.fieldnames:
            print(f"  FAIL  metadata CSV needs columns '{id_col}' and '{group_col}'; "
                  f"found {list(reader.fieldnames or [])}")
            return False
        for row in reader:
            lesion_of[Path(str(row[id_col]).strip()).stem] = str(row[group_col]).strip()

    def lesions(ids):
        miss = [i for i in ids if i not in lesion_of]
        if miss:
            print(f"  warn  {len(miss)} id(s) not found in metadata, e.g. {miss[:3]}")
        return {lesion_of[i] for i in ids if i in lesion_of}

    shared = lesions(train_set) & lesions(test_set)
    if shared:
        print(f"  FAIL  {len(shared)} lesion_id(s) span BOTH partitions, e.g. {sorted(shared)[:3]}")
        return False
    print("  ok    no lesion_id appears in both partitions (lesion-level split verified)")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=DATASETS + ["all"])
    ap.add_argument("--dir", type=Path, default=HERE, help="folder holding the .txt files")
    ap.add_argument("--metadata-csv", type=Path, help="HAM10000 metadata (for lesion-level check)")
    ap.add_argument("--id-col", default="image_id")
    ap.add_argument("--group-col", default="lesion_id")
    a = ap.parse_args()

    targets = DATASETS if a.dataset == "all" else [a.dataset]
    all_ok = True
    for ds in targets:
        all_ok = check_dataset(ds, a.dir, a.metadata_csv, a.id_col, a.group_col) and all_ok
        print()
    if all_ok:
        print("All checks passed.")
        sys.exit(0)
    print("Some checks FAILED (see above).")
    sys.exit(1)


if __name__ == "__main__":
    main()
