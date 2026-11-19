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

`experiment_gram.py` runs one resolution at a time. Both 224 and 512 use bilinear resize, EXIF orientation and ImageNet normalization. Within each batch, one original forward is shared across four global perturbations.

* `rigid_noise` adds Gaussian noise in normalized space with the original per-image seed policy and default sigma 0.05.
* `gaussian_blur` uses sigma 1 pixel at 224 and scales it by 512/224 at 512. The kernel radius is rounded up from three times sigma.
* `down_up` uses antialiased bilinear resizing to 75% of the input size and back.
* `wavelet` attenuates the existing one-level Haar detail by 50%.

Blur, resampling and Wavelet operate on RGB pixels before normalization. For each image and view, `scores.csv` records final CLS cosine distance, the 12 block CLS cosine distances, Patch Gram cosine and L1, and Block CLS Gram cosine, L1 and norm drop. Gram comparisons use the strictly upper-triangular pairs. Norm drop is `1 - ||G_perturbed|| / ||G_original||` on the 66 Block CLS pairs. The final CLS comes after the model's final LayerNorm. Block CLS tokens come from each block before that separate final LayerNorm. The matrices stay in memory for scoring within the batch.

```bash
python experiment_gram.py --name gram_global_224 --size 224 --samples all --batch-size 64 && \
python experiment_gram.py --name gram_global_512 --size 512 --manifest results/gram_global_224/manifest.csv --batch-size 64 && \
python analysis_gram.py --run results/gram_global_224 results/gram_global_512
```

For a smoke test use distinct run names, `--samples 2` on the first run, and `--batch-size 1`. Analysis accepts one run or a matched 224 and 512 pair. It saves per-generator and macro AUROC, every score's difference from the final CLS on that view in `delta_vs_final_cls.csv`, and `figures/macro_auroc.png`. A matched pair writes `resolution_comparison.csv` to the larger run. Previous single-Gaussian runs still use their saved Block CLS Gram archive for the original figures.

To evaluate the official RIGID protocol on the same image list, use the original DINOv2 ViT-L/14 backbone via Torch Hub, 224 bilinear resize, ImageFolder-style RGB decoding, and seed 100 by default. First use requires access to GitHub and the DINOv2 weights. The exact sampled noise can vary with batch size or device.

```bash
python experiment_gram.py --name rigid_official --size 224 --official-rigid \
  --manifest results/gram_global_224/manifest.csv --batch-size 64 && \
python analysis_gram.py --run results/gram_global_224 results/gram_global_512 --rigid-run results/rigid_official
```

The combined analysis writes `official_rigid_comparison.csv` alongside the resolution comparison.
