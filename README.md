# KCG-SME: Kinetic Chain Graph Modeling for Striking Motion Evaluation

KCG-SME is a deep learning framework for evaluating striking motion quality with explicit biomechanics grounding. It formulates striking motions as a chain-structured graph where nodes represent key body segments and edges encode inter-segment interactions and force transmission along the kinetic chain. Temporal modeling over the full motion cycle captures long-range dependencies in motion execution. While badminton striking motions are used as the case study, the framework generalizes to other kinetic-chain-dominated movements such as tennis strokes and baseball pitching.

---

## Repository Overview

- `models/kcg_sme_model.py`: KCG-SME model (chain-structured GCN + temporal Transformer + classifier).
- `train/`: Training entrypoints and configuration.
- `test/`: Evaluation scripts and metrics/plots.
- `data_processed/`: Preprocessing scripts for dataset generation.
- `app/skill_visualizer.py`: PyQt5 app for signal + skeleton visualization and prediction summary.

---

## Environment Setup

```bash
pip install torch h5py numpy pandas scipy matplotlib scikit-learn pyqt5
```

---

## Data Preparation

Dataset: MultiSenseBadminton
- Download: https://doi.org/10.6084/m9.figshare.c.6725706.v1
- Place raw data under `Data_Archive/`

Generate HDF5 files:

| Script | Description | Output |
|------|------|------|
| `python data_processed/data_preprocessing.py` | Merge all sensor streams, keep Forehand Clear only, and generate the 38-dim feature matrix. | `data_processed/data_processed_allStreams_60hz_onlyForehand_skill_level.hdf5` |
| `python data_processed/data_preprocessing_skeleton.py` | Extract `pns-joint/global-position` for the GUI 3D skeleton. | `data_processed/data_processed_allStreams_60hz_onlyForehand_skeleton_skill_level.hdf5` |

---

## Training and Evaluation

| Scenario | Command | Notes |
|------|------|------|
| Train KCG-SME (default) | `python train/train.py` | Random sample split 70/20/10, checkpoint saved to `outputs/checkpoints/kcg_sme_best.pt`. |
| Evaluate only | `python test/test.py --checkpoint outputs/checkpoints/kcg_sme_best.pt` | Uses the same split and feature configuration as training. |
| GUI visualization | `python app/skill_visualizer.py` | Auto-loads `train/config.py`. |

You can override config via env vars, e.g.:

```bash
KCG_SME_EPOCHS=50 python train/train.py --subject-split
```

Training uses early stopping by default (`patience=20`, based on val accuracy). Adjust `early_stopping_patience` in `train/config.py` if needed.

---

## Splits and Feature Subsets

### Subject-level split / k-fold

| Command | Notes |
|------|------|
| `python train/train.py --subject-split` | Subject-level split (same player never appears in multiple splits), avoids identity leakage. |
| `python train/train.py --kfold 5` | 5-fold cross-validation (supports `--subject-split`). Logs train/val only. |
| `python test/test.py --subject-split` | Uses the same subject split as training. |

### Node Feature Ablation

`train/config.py` defines a 7-node to 38-dim mapping. Mask columns with `--feature-subset`:

| Key | Node set | Column indices |
|-----|---------|--------------|
| `node13` | {1,3} | `[0,1,2,6,7,8,9]` |
| `node123` | {1,2,3} | `[0..9]` |
| `node1234` | {1,2,3,4} | `[0..18]` |
| `node12345` | {1,2,3,4,5} | `[0..26]` |
| `node123456` | {1,2,3,4,5,6} | `[0..34]` |
| `node1234567` / `all` | all nodes | `[0..37]` |

Example:

```bash
python train/train.py --feature-subset node1234
python test/test.py --feature-subset node1234 --checkpoint outputs/checkpoints/kcg_sme_best.pt
```

---

## Module Ablation (KCG-SME)

Use `script/run_kcg_sme_ablation.py` to compare variants with consistent splits and metrics.

```bash
python script/run_kcg_sme_ablation.py \
    --epochs 200 \
    --batch-size 32 \
    --variants full,no_gcn,no_transformer,no_pos_cls \
    --feature-subsets all
```

Output JSON: `outputs/logs/kcg_sme_ablation.json`

---

## GUI Visualization

```bash
python app/skill_visualizer.py
```

Features:
- EMG and insole/joint curves with consistent styling.
- 2.5 s skeleton animation (auto-play, pausable).
- Evaluation summary with true/predicted labels and class probabilities.

---

## License

Add your license here.
