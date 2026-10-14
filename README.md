# ENG2440 Assignment 2: Whole-Tumour Segmentation (MSD Task01 BrainTumour)

This repo trains a 3D U-Net (PyTorch) to segment the whole tumour (all labels > 0) in 4-channel brain MRI. It compares two patch samplers, uniform and tumour-aware, and evaluates the frozen pipeline once on a held-out test split. Everything runs from one notebook, `medass2.ipynb`.

## Setup

```bash
conda env create -f environment.yml
conda activate medass2
python -m ipykernel install --user --name medass2 --display-name medass2
jupyter lab            # open medass2.ipynb, select the "medass2" kernel
```

Pinned versions (from the environment used for the submitted run):

| Package | Version |
|---|---|
| Python | 3.11.16 |
| PyTorch | 2.11.0+cu128 (CUDA 12.8, cuDNN 9.19) |
| nibabel | 5.4.2 |
| NumPy | 2.4.6 |
| pandas | 3.0.6 |
| SciPy | 1.17.1 |
| Matplotlib | 3.11.2 |
| JupyterLab / ipykernel | 4.6.4 / 7.4.0 |
| tabulate | 0.10.0 (needed by `DataFrame.to_markdown`) |

Run the notebook with the repo root as the working directory. The split CSV, `init_weights.pt`, `runs/` and `figures/` are all opened through relative paths.

## Data

1. Download **Task01_BrainTumour** from the Medical Segmentation Decathlon (<http://medicaldecathlon.com/>, direct link: <https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar>, about 7.6 GB).
2. Extract it in the repo root:
   ```bash
   tar -xf Task01_BrainTumour.tar
   ```
   This gives `Task01_BrainTumour/{dataset.json, imagesTr/, labelsTr/, imagesTs/}`.
3. The split file `assignment2_split(in).csv` is in the repo. It lists all 484 labelled cases (`imagesTr`/`labelsTr`) with `case_id`, `split` (`train` / `validation` / `test`), `image_relpath` and `label_relpath`. The unlabelled `imagesTs/` is not used.

**Paths to change.** Both paths are set in the first code cell:

```python
DATA_ROOT = Path("Task01_BrainTumour")     # <- the one line to change if the dataset lives elsewhere
CACHE_DIR = Path("cache_preprocessed")     # preprocessed .npy cache (~4.9 GB); change if disk space is tight
```

No other paths need editing. The dataset, the `.tar`, `cache_preprocessed/` and all `*.nii.gz` / `*.npy` files are git-ignored.

## Exact run order

Run the cells top to bottom in a fresh kernel. Several later cells redefine earlier ones (for example `PatchDataset`, `augment`, `train` and the loss), and the last definition is the one that is used. In the table below, cell numbers count from 0 at the title markdown cell.

| Cells | Section | What it does | Writes |
|---|---|---|---|
| 1 | Setup | Imports, `DATA_ROOT`, `CACHE_DIR` | none |
| 2–7 | A. Inspect | Load and plot BRATS_001, check channel order, orientation and spacing | none |
| 8–10 | Split checks | Assert 484 rows, every file present, no duplicates, no overlap between splits | none |
| 11 | Volumes | Tumour volume (mL) per train/val case | none |
| 12–14 | Geometry | Reference shape, affine and spacing; image/label consistency over all 484 cases | none |
| 15–16 | Preprocessing demos | Nearest vs linear label resampling; per-channel brain z-score | none |
| 17–20 | Crop box | Global brain bounding box, crop/uncrop round trip, checks that no brain or tumour is cut (train/val and test) | none |
| 21–23 | Augmentation and `preprocess_case` | L-R flip plus per-channel intensity scale/shift | none |
| 24–26 | Cache | Cell 24 caches 3 cases, cell 25 verifies them, **cell 26 caches all 411 train+val cases** | `cache_preprocessed/*.npy`, `crop_box.json` |
| 27–31 | Data loader | Patch dataset and loader; cell 30 sets the final `PATCH_SIZE = (64, 64, 64)`; cell 31 compares 96³ and 64³ patches | none |
| 32 | Model | `UNet3D(base=16)`, shape check | none |
| 33 | Trial run | Creates `init_weights.pt` if missing; 1 epoch × 20 iterations | `runs/trial_uniform/` |
| 34 | Uniform baseline | Fixes 100 validation patches (needed by cell 35); trains uniform for 50 × 200 iterations | `runs/uniform/` (overwritten by cell 35) |
| 35 | **Main training** | Uniform and tumour-aware (`p_tumour=0.5`) samplers, **200 epochs × 200 iterations each** | `runs/uniform/`, `runs/tumour/` (`best.pt`, `last.pt`, `history.json`) |
| 36 | Curves | Training curves | `figures/training_curves.png` |
| 37–39 | Full-volume validation | Sliding window (64³, 50 % overlap); uniform vs tumour on 73 validation cases; Wilcoxon test | `runs/val_full_volume_metrics.csv` |
| 40–41 | Post-processing | Connected-component rules on validation (`FINAL_SAMPLER = "tumour"`) | `runs/val_cc_rule_metrics.csv` |
| 42 | **Test (run once)** | Frozen pipeline on 73 test cases | `runs/test_metrics.csv`, `runs/test_evaluated.flag` |
| 43–45 | Test reporting | Tables, volume agreement | `runs/test_before_after_cc.csv`, `runs/test_volume_per_case.{csv,md}`, `figures/test_volume_*.png` |
| 46–48 | G. Case review | Best, median and worst cases plus one extra case; error maps; BRATS_376 component analysis | `runs/section_g_cases.csv`, `figures/section_g_error_maps.png` |
| 49–50 | Bonus | Missing-modality stress test | `runs/bonus_missing_modality_test.csv` |
| 51 | Bonus | Sliding-window overlap 0 / 0.25 / 0.5 and seam analysis | `runs/bonus_overlap_test.csv`, `figures/bonus_overlap_seams.png` |

Notes:
- **Test guard.** Cell 42 asserts that `runs/test_evaluated.flag` does not exist. With the flag present, "Run All" stops at cell 42. To review results without re-testing, run cells 1–41 and then 43 onwards; those cells read `runs/test_metrics.csv`. For a full reproduction from scratch, delete the flag first.
- `G_CASES` in cell 47 is hard-coded from the output of cell 46.
- Cells 28–29 only print diagnostics (GPU availability, cache count).

## Seeds

| Seed | Value | Where it is used |
|---|---|---|
| `SEED` | 42 | `torch.manual_seed` and `np.random.seed` in `train()`; `PatchDataset` RNG; `DataLoader` generator; creating `init_weights.pt`; model init in cell 32; augmentation demo (cell 21); tumour-voxel index subsampling |
| `VAL_SEED` | 123 | The 100 fixed validation patches used for model selection |
| per-worker | `torch.initial_seed() % 2**32` | `worker_init_fn` re-seeds each DataLoader worker's NumPy RNG |

Both samplers start from the same `init_weights.pt`, which is only created when the file is missing. They also share the same validation patches and training budget.

Training is not bit-exact across runs or machines. The code does not set `torch.backends.cudnn.deterministic` or `benchmark`, uses AMP (fp16 autocast), and loads data with `num_workers=4`.

## Hardware & runtime

- The code uses a CUDA GPU when one is available and falls back to CPU otherwise. AMP is used only on CUDA.
- Training: batch size 4, patch 64³, Adam with LR 3e-4, `num_workers=4`.
- GPU used: NVIDIA RTX 5000 Ada Generation, 32 GB
- Peak GPU memory during training: about 0.7 GB
- Training per sampler, 200 × 200 = 40,000 iterations at batch size 4 (cell 35): about 20.3 min per run (uniform and tumour)
- Validation full-volume comparison (cell 38, 73 cases × 2 models): about 55 s
- Test inference (cell 42): about 0.18 s per case for the sliding window (64³, 50 % overlap); computing the metrics, including HD95, adds about 2 s per case
- Disk: dataset ≈ 7.1 GB extracted (plus 7.6 GB for the `.tar`), cache ≈ 4.9 GB

The `sec_per_iter` values recorded for each epoch are in `runs/<run>/history.json`.

## Outputs

**Checkpoints and logs** (`runs/`)
- `init_weights.pt` (repo root): shared initial weights
- `runs/{trial_uniform,uniform,tumour}/best.pt`: best patch-validation Dice, saved as `{"model", "epoch", "val_dice"}`
- `runs/{…}/last.pt`: final epoch
- `runs/{…}/history.json`: per-epoch train loss, validation loss, validation Dice, s/iter and the hyperparameters

**Metrics** (`runs/`)
- `val_full_volume_metrics.csv`: per validation case, uniform vs tumour
- `val_cc_rule_metrics.csv`: per validation case for each connected-component rule (`none`, `largest`, `min_0.1`, `min_0.5`)
- `test_metrics.csv`: per test case for rules `none` (official) and `min_0.5`
- `test_before_after_cc.csv`: summary table
- `test_volume_per_case.{csv,md}`
- `section_g_cases.csv`
- `bonus_missing_modality_test.csv`, `bonus_overlap_test.csv`

**Figures** (`figures/`)
- `training_curves.png`, `test_volume_agreement.png`, `test_volume_scatter.png`, `section_g_error_maps.png`, `bonus_overlap_seams.png`

**Headline results** (official rule `none`, 73 test cases)

| Metric | Value |
|---|---|
| Final model | Tumour-aware sampler (`p_tumour = 0.5`), `best.pt` from epoch 168 |
| Best patch val Dice: uniform / tumour | 0.910 / 0.911 |
| Full-volume val Dice, 73 cases: uniform / tumour (mean ± SD) | 0.886 ± 0.085 / 0.901 ± 0.057 (paired Wilcoxon p = 0.0001; tumour-aware better in 55/73) |
| Test Dice (mean ± SD, median) | 0.906 ± 0.056 (0.925) |
| Test IoU | 0.832 ± 0.087 (0.860) |
| Test sensitivity | 0.894 ± 0.073 (0.914) |
| Test HD95 (mm) | 5.72 ± 7.17 (3.00) |
| Abs. volume error | 8.76 ± 8.13 mL (median 7.13); mean abs. 9.5 % |
| Empty predictions | 0 |
| Post-processing | None. A < 0.5 mL component filter was evaluated: Dice +0.001, HD95 worse |
