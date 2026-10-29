# WET-Net

Reference implementation of **WET-Net: A Wavelet-Evidential Topology-Aware Network for Robust
and Lightweight Skin Lesion Segmentation**.

This repository accompanies the revised manuscript submitted to *Multimedia Tools and
Applications*. It was written in response to the reviewers' comments, and it is organised so
that each contested claim can be checked rather than taken on trust.

---

## ⚠Read this before using the numbers

 things in this repository are **measured**, 

| Quantity | Status |
| --- | --- |
| Parameter counts, GFLOPs, latency | Measured here. Run `scripts/complexity_table.py`. |
| Mathematical properties (perfect reconstruction, invertibility, metric identities, topology detection) | Machine-checked here. Run `scripts/verify_claims.py` and `pytest`. |
| **Segmentation accuracy on ISIC2018 / ISIC2017 / HAM10000 / PH2** | **NOT in this repository.** It requires training on the real datasets, which are not redistributed here. Run `scripts/train.py`. |




Measured for the architecture exactly as described in the manuscript (four LRRE blocks,
widths 16/32/64/96, one depthwise + pointwise + 1×1-shortcut per stage, 256×256 input):

| | Manuscript | Measured here |
| --- | --- | --- |
| Total parameters | 0.312 M | **0.074 M** |
| GFLOPs | 0.184 | **0.185** (with `decoder_out=24, tar_hidden=24, tar_stride=4`) |
| Encoder-only parameters | 0.16 M | **0.020 M** |

The GFLOPs figure reproduces to within 0.6 %. The parameter count does not: the described
architecture yields roughly a quarter of the reported figure. A search over channel widths,
blocks per stage and TAR resolution (`scripts/match_budget.py`) found **no configuration that
reaches 0.312 M parameters and 0.184 GFLOPs simultaneously** — the closest to the parameter
target, widths (16, 32, 64, 384), costs 0.272 GFLOPs.

Either the manuscript omits an architectural detail (most plausibly more than one LRRE block
per stage, or wider channels than stated) or one of the two figures is wrong. `blocks_per_stage`
and `widths` are exposed in `WETNetConfig` so the intended configuration can be reproduced once
it is known. **This must be resolved before the revision is submitted**, because a reviewer
running this code will hit it immediately.

---

## Install

```bash
pip install -r requirements.txt
pytest -q            # 38 tests, ~2 s
```

`gudhi` is strongly recommended: without it the persistence backend falls back to a slower
union-find implementation whose H1 pairing is approximate.

## Quick start

```bash
# efficiency columns of Tables III-VIII, all measured
python scripts/complexity_table.py

# machine-check the mathematical claims of Sections III-C, III-B and IV-C
python scripts/verify_claims.py

# end-to-end sanity check + the topological-loss A/B, on synthetic data
python scripts/smoke_test.py --epochs 16 --size 96

# train on a real dataset
python scripts/train.py --config configs/isic2018.yaml
```

## Reproducing each table

Every table comes from the same entry point with different flags.

| Table | Command |
| --- | --- |
| III — encoder | `--encoder lrre \| resnet18 \| resnet34 \| mobilenetv3_small` |
| IV — modules | `--bridge doubleconv`, `--no-tar` |
| V — bridge | `--bridge web \| pdcp \| aspp \| spp` |
| VI — fusion rule | `--fusion evidential \| se \| static \| uniform` |
| VII — wavelet basis | `--wavelet db2 \| db5 \| db7`, `--levels 1 \| 2 \| 3` |
| VIII — comparison | `configs/{isic2018,isic2017,ham10000,ph2}.yaml` |

Splits are index files (`splits/*.txt`), not shuffle seeds, so a run is reproducible
rather than merely re-derivable.

## What answers which reviewer comment

| Comment | Where |
| --- | --- |
| **R1-5, R1-7** — the exact-reconstruction claim is inconsistent with re-weighting | `wetnet/layers/dtcwt.py`. The filter bank is a **paraunitary lattice**, so perfect reconstruction holds for *any* parameter value, including after training. `tests/test_dtcwt.py` asserts the two properties the revision actually claims: losslessness, and that the identity is representable at uniform weights. |
| **R1-7** — no ablation isolating the evidential mechanism | `wetnet/layers/web.py`, `fusion=` selects uniform / static gate / SE gate / evidential over an identical decomposition. `tests/test_web.py::test_only_evidential_emits_uncertainty` shows what the gates cannot do. |
| **R1-8** — release the code | This repository. |
| **R3** — encoder ablation mixes counting conventions | `WETNet.parameter_report()` returns encoder-only *and* whole-network counts separately; `scripts/complexity_table.py` prints both in every row. |
| **R4-3** — why 1×1, what stride | `wetnet/layers/lrre.py` module docstring, with the cost arithmetic and the aliasing argument for keeping downsampling out of the convolutions. |
| **R4-4, R4-5** — GELU and Softplus | `lrre.py` and `web.py` docstrings, with formulas. `EvidentialHead` also carries a float32 underflow floor — `nn.Softplus` returns exactly 0 below about x = −104, which would silently reintroduce the ReLU dead-band the activation was chosen to avoid. |
| **R4-6** — check the Dice formula | `wetnet/metrics.py` implements both the confusion and set forms; `tests/test_metrics.py` asserts they agree to machine precision. It also separates `iou_lesion` from `iou_macro`, the labelling ambiguity the check exposed. |

## Layout

```
wetnet/
  model.py              WETNet + WETNetConfig
  complexity.py         parameter / MAC / latency accounting
  losses.py             BCE + Dice + evidential KL + persistence
  metrics.py            mIoU, Dice, Acc, Sp, Se, HD95, ECE
  data.py               dataset readers + synthetic lesions
  layers/
    dtcwt.py            paraunitary lattice, dual-tree transform
    web.py              Wavelet-Evidential Bridge + fusion variants + bridge baselines
    tar.py              PersLay + cross-attention refinement
    persistence.py      differentiable cubical persistent homology
    lrre.py             LRRE block, encoder, decoder, baseline encoders
scripts/                train, evaluate, complexity_table, verify_claims, smoke_test, match_budget
tests/                  38 tests
results/                measured output, regenerated by the scripts
```

## Citation

```bibtex
@article{arora2026wetnet,
  title   = {WET-Net: A Wavelet-Evidential Topology-Aware Network for Robust and
             Lightweight Skin Lesion Segmentation},
  author  = {Arora, Amit Kumar and Kaur, Amanpreet},
  journal = {Multimedia Tools and Applications},
  year    = {2026}
}
```

## License

MIT — see `LICENSE`.
