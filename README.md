# sem9
The model builds on the Senseiver architecture: <https://github.com/OrchardLANL/Senseiver>
The dataset is built on a modified version of <https://github.com/hietwll/LBM_Taichi/tree/master>
## Setup

```bash
cd ws
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Generate dataset

```bash
python smoke_simulator.py
```

## Train Model

```bash
python train.py
```

## Evaluate

```bash
python evaluate.py
```
