# HumanLift Stage 3 Runbook

This document is for the HumanLift pipeline where stage 2 produces 81 orthographic-like human views and stage 3 reconstructs a Gaussian scene from those views.

## 1. Prepare input from stage 2 frames

Expected stage-2 output:

- `frames_480P_<ref_name>/frame_00.png ... frame_80.png`

Convert those frames into a stage-3 case:

```bash
python 3-gs_recon/tools/prepare_humanlift_case.py \
  --frames-dir /path/to/frames_480P_sample_processed \
  --output-dir 3-gs_recon/data/sample_stage3 \
  --overwrite
```

That command creates:

- `3-gs_recon/data/sample_stage3/images/lgt0_r_0000.png ... lgt0_r_0080.png`
- `3-gs_recon/data/sample_stage3/transforms_train.json`
- `3-gs_recon/data/sample_stage3/transforms_test.json`
- `3-gs_recon/data/sample_stage3/metadata.json`

## 2. Smoke test before full run

Always run a short save-point test first.

```bash
cd 3-gs_recon
python train.py \
  -s data/sample_stage3 \
  -m output/sample_stage3_smoke \
  --iterations 2000 \
  --test_iterations 2000 \
  --save_iterations 2000 \
  --checkpoint_iterations 2000 \
  --disable_viewer
```

If this fails, do not start the long run.

## 3. HumanLift densification smoke test

The default 3DGS pruning settings are too aggressive for HumanLift stage-2 synthetic views.
Before running the full experiment, verify densification with a HumanLift-specific smoke test:

```bash
cd 3-gs_recon
python train.py \
  -s data/sample_stage3 \
  -m output/sample_stage3_smoke_densify \
  --iterations 2500 \
  --test_iterations 2000 2500 \
  --save_iterations 1600 1700 2500 \
  --checkpoint_iterations 1600 1700 2500 \
  --prune_min_opacity 0.0001 \
  --opacity_prune_from_iter 3000 \
  --min_keep_gaussians 20000 \
  --disable_viewer
```

Recommended interpretation:

- if point count still collapses before `2500`, do not start the full run
- if this smoke test finishes and renders start to form a body instead of a blob, proceed to the full run

## 4. Full experiment

Once the smoke test is stable, run a save-point before and after the first densification window.

```bash
cd 3-gs_recon
python train.py \
  -s data/sample_stage3 \
  -m output/sample_stage3_full \
  --iterations 30000 \
  --test_iterations 2000 5000 10000 20000 30000 \
  --save_iterations 2000 5000 10000 20000 30000 \
  --checkpoint_iterations 2000 5000 10000 20000 30000 \
  --disable_viewer
```

Recommended behavior:

- save at `2000` so you always have a pre-densification checkpoint
- save at `5000` so you can compare early densification behavior
- if training crashes after densification starts, resume from `chkpnt2000.pth` and debug from there

## 5. Resume from checkpoint

```bash
cd 3-gs_recon
python train.py \
  -s data/sample_stage3 \
  -m output/sample_stage3_resume \
  --iterations 30000 \
  --test_iterations 5000 10000 20000 30000 \
  --save_iterations 5000 10000 20000 30000 \
  --checkpoint_iterations 5000 10000 20000 30000 \
  --start_checkpoint output/sample_stage3_full/chkpnt2000.pth \
  --disable_viewer
```

## 6. Render checkpoints

```bash
cd 3-gs_recon
python render.py \
  -m output/sample_stage3_full \
  -s data/sample_stage3 \
  --iteration 5000 \
  --skip_test
```

Rendered images are written under:

- `output/sample_stage3_full/train/ours_<iter>/renders`
- `output/sample_stage3_full/train/ours_<iter>/gt`

## 7. Current implementation notes

This repo includes HumanLift-specific stage-3 compatibility patches:

- synthetic 81-view camera fallback when `pytorch3d` is unavailable
- optional import for `diff_gaussian_rasterization`
- SH feature layout normalization for gsplat-based rendering

Those patches are meant to make stage 3 reproducible on the same runtime used for HumanLift stage 1 and stage 2.
