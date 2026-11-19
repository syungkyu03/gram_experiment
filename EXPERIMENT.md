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
