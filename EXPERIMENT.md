# Experiment views

Default views (one score row per image and view):

* `rigid_noise`: 224×224 bilinear resize, ImageNet normalization, then Gaussian noise with standard deviation `--rigid-noise-std` (default 0.05) and no clipping.
* `rigid_local_noise`: same 224 preprocessing and normalized-space noise applied only to a random rectangular region. This is our extension, not an official RIGID method.
* `rigid_wavelet`: 224×224 bilinear resize, then the same pixel-space Haar high-frequency attenuation as `wavelet`, followed by normalization. This is our extension, not an official RIGID method or WaRPAD.
* `noise`, `local_noise`, `wavelet`: previous experiment's 512×512 bicubic resize by default; noise is applied in pixel space and clipped. `--size` changes this branch only.
* `vae`: optional previous reconstruction view; requires `--vae-path` and including `vae` in `--transforms`.

All views share the DINOv3 backbone. `rigid_noise` with `cls_cosine_change` uses the RIGID score definition but does not reproduce the official backbone or RNG sequence. EXIF orientation is applied. Per-image noise is seeded from the absolute path, seed, and view, so moving files changes perturbations. Local A includes every patch touching the rectangle; B is the rest. `patch_change_b` is the mean patch L2 difference over B, matching the spatial subset of `gram_bb`.

```bash
python experiment.py --name comparison_smoke --samples 2 --batch-size 1 --device cuda:0
python analyze.py --run results/comparison_smoke --bootstrap 0
```

To resume, use the same experiment arguments plus `--resume`. Existing runs with older score schemas cannot be resumed with this version. Use a new experiment name. Selecting `--transforms` overrides defaults completely. Bootstrap analysis can take a long time; `--bootstrap 0` skips it.

## Left–right perturbation experiment

`experiment_lr.py` uses the Ring input setup (512×512 bicubic resize and DINOv3 ViT-B/16). It runs the original image and two versions with the same 128×128 Gaussian noise tensor applied separately to symmetric locations. Both readout regions stay unmodified and contain 32 patches. In patch coordinates the left mask and readout occupy columns 4–11 and 0–3, and the right mask and readout occupy columns 20–27 and 28–31. All four regions occupy rows 12–19. Readout distances from a mask are 1–4 patches on the same side and 17–20 across the image.

`scores.csv` stores one row per image with four mean Patch L2 responses. The first letter of each response identifies the perturbation side and the second identifies the readout side. `near_patch` is the geometric mean of LL and RR, and `far_patch` is the geometric mean of LR and RL. `lr_cross_ratio` divides `far_patch` by `near_patch`. The analysis compares all three scores on the same finite rows and records omissions in `audit.json`.

```bash
python experiment_lr.py --name lr_smoke --samples 2 --batch-size 1 --device cuda:0 && python analysis_lr.py --run results/lr_smoke
```

Use `--samples all` and a new run name for the full run. `analysis_lr.py` writes per-generator AUROC, `lr_comparison.csv` with differences from the far-only score, and `figures/auroc_lr.png`.

## Whole-matrix Gram experiment

`experiment_gram.py` runs one resolution at a time. Both 224 and 512 use bilinear resize and ImageNet normalization. The second forward adds global Gaussian noise in normalized space with the existing `rigid_noise` seed policy and default sigma 0.05.

The primary scores are final CLS cosine distance, final Patch Gram cosine distance, and Block CLS Gram cosine distance. Both Gram scores compare the strict upper triangles with every pair included. Gram L1 and direct CLS cosine distance for each block are saved as diagnostic scores. The final CLS comes after the model's final LayerNorm, while every CLS in the Block Gram comes from the block output before that final LayerNorm.

`scores.csv` stores one row per image. `block_cls_grams.npy` stores the float32 upper triangles in manifest order with shape `[images, 2, block_pairs]`. The second axis is original then noisy, and the pair order follows `triu_indices` with offset 1. This takes about 53 MB for 100,000 images with the 12-block model. Patch Gram matrices are reduced to scores in memory.

```bash
python experiment_gram.py --name gram_224 --size 224 --samples all --batch-size 64 && \
python experiment_gram.py --name gram_512 --size 512 --samples all --batch-size 64 && \
python analysis_gram.py --run results/gram_224 results/gram_512
```

For a smoke test use distinct run names, `--samples 2`, and `--batch-size 1`. Analysis accepts one run or a matched 224 and 512 pair. It audits each run and saves per-generator AUROC, an AUROC heatmap, and Real/Fake Block CLS Gram heatmaps. The paired invocation checks the manifests and settings before printing a macro AUROC comparison.
