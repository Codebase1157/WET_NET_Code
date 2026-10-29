# WET-Net

## Repository structure

```text
.
├── assets/
├── configs/
│   ├── ham10000.yaml
│   ├── isic2017.yaml
│   ├── isic2018.yaml
│   ├── ph2.yaml
│   └── synthetic.yaml
├── results/
├── scripts/
│   ├── complexity_table.py
│   ├── evaluate.py
│   ├── match_budget.py
│   ├── smoke_test.py
│   ├── train.py
│   └── verify_claims.py
├── splits/
│   ├── ham10000_test.txt
│   ├── ham10000_train.txt
│   ├── isic2017_test.txt
│   ├── isic2017_train.txt
│   ├── isic2018_test.txt
│   ├── isic2018_train.txt
│   ├── make_splits.py
│   ├── ph2_test.txt
│   ├── ph2_train.txt
│   ├── README.md
│   └── verify_splits.py
├── tests/
├── wetnet/
│   ├── layers/
│   ├── complexity.py
│   ├── data.py
│   ├── losses.py
│   ├── metrics.py
│   └── model.py
├── LICENSE
├── README.md
└── requirements.txt
```

## Installation

```bash
python -m pip install -r requirements.txt
```

## Run

Run tests:

```bash
pytest -q
```

Run the main scripts:

```bash
python scripts/complexity_table.py
python scripts/verify_claims.py
python scripts/smoke_test.py --epochs 16 --size 96
```

Train using a dataset configuration:

```bash
python scripts/train.py --config configs/isic2018.yaml
```

Available configurations:

- `configs/isic2018.yaml`
- `configs/isic2017.yaml`
- `configs/ham10000.yaml`
- `configs/ph2.yaml`
- `configs/synthetic.yaml`

Edit the image, mask, and index paths in the selected YAML configuration to match the local dataset layout before training.

## License

MIT License. See `LICENSE`.
