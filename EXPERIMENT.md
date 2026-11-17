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
