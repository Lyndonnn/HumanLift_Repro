# HumanLift Reproduction Guide

This repository is a reproduction workspace for:

- Original code: https://github.com/IGLICT/HumanLift
- Paper/project: HumanLift: Single-Image 3D Human Reconstruction with 3D-Aware Diffusion Priors and Facial Enhancement
- Reproduction repo: https://github.com/Lyndonnn/HumanLift_Repro

## 1. Repository Policy

Do not edit the official repository directly. Keep this repository as your working reproduction repo:

- `origin`: `https://github.com/Lyndonnn/HumanLift_Repro.git`
- `upstream`: `https://github.com/IGLICT/HumanLift.git`

The official repository should remain the provenance source. Keep the original `LICENSE`, README citation, and acknowledgements. Add reproduction notes in separate files or clearly marked sections so your changes are auditable.

Recommended branch layout:

- `main`: reproducible baseline plus stable fixes.
- `colab`: optional notebooks and Colab-specific scripts.
- `exp/<name>`: one experiment per branch, for example `exp/wan-rgb-argparse` or `exp/3gs-small-iters`.

Common local sync commands:

```bash
git status
git pull --rebase origin main
git fetch upstream
git log --oneline --decorate -5
git add .
git commit -m "Add reproduction setup notes"
git push origin main
```

To pull official updates later:

```bash
git fetch upstream
git merge upstream/main
git push origin main
```

If the merge is noisy, prefer a new branch:

```bash
git switch -c sync/upstream-YYYYMMDD
git merge upstream/main
```

## 2. What Should Not Be Committed

Do not commit these to GitHub:

- Wan2.1 base model weights.
- HumanLift fine-tuned LoRA/checkpoints.
- SMPL/SMPL-X licensed model files.
- Training datasets.
- Colab outputs, videos, Gaussian checkpoints, `.ply` exports, and generated image folders.

Use Google Drive, Hugging Face private repos, or release assets for large artifacts. Git should contain code, configs, scripts, notebooks, and small reproducibility metadata.

## 3. Colab Workflow

Colab should pull code from your repo every run. Do not edit code permanently inside Colab unless you commit and push it back.

Minimal Colab bootstrap:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
cd /content
git clone https://github.com/Lyndonnn/HumanLift_Repro.git
cd HumanLift_Repro
git pull --rebase origin main
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

For later runs:

```bash
cd /content/HumanLift_Repro
git pull --rebase origin main
```

After a successful Colab code change:

```bash
git status
git add path/to/changed_file.py
git commit -m "Describe reproducibility fix"
git push origin main
```

If Colab has no GitHub credential, push from local instead: download or copy the changed file back to this workspace, commit locally, then push.

## 4. Reproduction Stages

The practical order is:

1. Run official inference with provided checkpoints on one sample.
2. Reproduce multi-view RGB and normal generation.
3. Reproduce 3D Gaussian reconstruction from generated RGBA views.
4. Only then attempt LoRA training, because the released training script has hard-coded author paths and expects large data.
5. Optional: run LHM-based animation after reconstruction assets are verified.

## 5. Stage 0: Inputs and Model Assets

Prepare these outside Git:

- A full-body single-person image, preferably clean background and visible limbs.
- Wan2.1-I2V-14B-720P base weights.
- HumanLift multi-view RGB LoRA/checkpoints from the official release link.
- HumanLift multi-view normal LoRA/checkpoints from the official release link.
- SMPL/SMPL-X assets required by the preprocess or LHM stack.

Suggested Drive layout:

```text
/content/drive/MyDrive/HumanLift/
  models/
    Wan2.1-I2V-14B-720P/
    humanlift_rgb/
    humanlift_normal/
    smplx/
  data/
    inputs/
    motion/
    multiview_rgb/
    multiview_normal/
    gs_ready/
  outputs/
    3gs/
    logs/
```

Use symlinks in Colab rather than copying large models:

```bash
ln -s /content/drive/MyDrive/HumanLift/models/Wan2.1-I2V-14B-720P 2-mv_gen/Wan2.1-I2V-14B-720P
```

## 6. Stage 1: SMPL-X / Pose Preprocess

The README command is:

```bash
cd 1-preprocess
python video2motion.py \
  --input_path ./images/2.jpg \
  --output_path ./motion \
  --visualize
```

Expected output is a motion folder containing rendered SMPL-X condition images. The multi-view scripts later expect the image and pose folders under:

```text
2-mv_gen/data/data/test/
2-mv_gen/data/output/test/
```

For a first reproduction, use exactly one test image and record:

- original image path
- processed image path
- output motion folder
- script command
- GPU type
- package versions

## 7. Stage 2: Multi-View Generation

Install the multi-view generation dependencies in Colab:

```bash
cd /content/HumanLift_Repro/2-mv_gen
pip install -r requirements.txt
pip install huggingface_hub Pillow numpy==1.23
```

The official README also requires PyTorch3D, mmcv-full, and mmhuman3d for SMPL condition images. Colab CUDA/PyTorch versions change, so verify `torch.__version__` and install matching wheels. If this becomes fragile, isolate preprocess in one notebook and Wan inference in another.

Prepare input folders:

```text
2-mv_gen/data/data/test/<image>.png
2-mv_gen/data/output/test/<image_name>/temp/smplx1/images/*.png
```

Then update/check these hard-coded values before running:

- `2-mv_gen/inference_wan_rgb.py`
  - `xxx = "test"`
  - `image_dir = f"data/data/{xxx}"`
  - `pose_base_dir = f"data/output/{xxx}"`
  - Wan2.1 base model path
  - RGB LoRA checkpoint folder/version/checkpoint id
- `2-mv_gen/inference_wan_normal.py`
  - dataset id such as `xxx`
  - generated RGB output folder used as conditioning
  - normal LoRA checkpoint folder/version/checkpoint id

Run:

```bash
cd /content/HumanLift_Repro/2-mv_gen
python inference_wan_rgb.py
python inference_wan_normal.py
```

Expected output is 81 frames for a full horizontal orbit. Save the exact output folder names in your experiment log.

## 8. Stage 3: Background Removal and 3DGS Dataset

The reconstruction stage expects transparent RGBA images resized/padded to `832x832`.

Target layout:

```text
3-gs_recon/data/test/
  images/
    lgt0_r_0000.png
    lgt0_r_0001.png
    ...
    lgt0_r_0080.png
  transforms_train.json
  transforms_test.json
  transforms_val.json
  intrinsics.txt
```

Important details:

- Keep file names sequential.
- Use alpha transparency after background removal.
- Do not overwrite the tracked JSON camera template unless you intentionally change camera sampling.
- For debugging, start with fewer training iterations.

Install reconstruction dependencies:

```bash
cd /content/HumanLift_Repro/3-gs_recon
pip install -r /content/HumanLift_Repro/2-mv_gen/requirements.txt
pip install gsplat plyfile natsort tensorboard
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install submodules/fused-ssim
```

Run a smoke test:

```bash
cd /content/HumanLift_Repro/3-gs_recon
python train.py -s data/test -m output/smoke --iterations 200 --disable_viewer
```

Run full reconstruction:

```bash
python train.py -s data/test -m output/test --disable_viewer
```

Default training is 30,000 iterations. The script saves at 7,000 and 30,000 by default.

## 9. Stage 4: Training the Multi-View Prior

Treat training as a second milestone, not the first reproduction step.

The released `2-mv_gen/train.sh` assumes:

- Multi-GPU machine via `CUDA_VISIBLE_DEVICES="4,5,6,7"`.
- Author-local dataset paths like `/home/jovyan/data2/...`.
- Pickle files inside each sample directory.
- Wan2.1 14B weights.
- A pretrained LoRA checkpoint.

The dataset reader in `train_wan.py` loads:

```text
<sample_dir>/<typea>.pkl
<sample_dir>/<typeb>.pkl
```

For RGB training, official defaults are:

```text
typea = frame_data_images
typeb = smplx_with_foot_wo_face_images
```

For normal training, official comments suggest:

```text
typea = frame_data_normals
typeb = frame_data_images
```

Before training on Colab, patch `train_wan.py` so `TextVideoDataset_onestage1` uses `--dataset_path` instead of the hard-coded `/home/jovyan/...` paths. Then use a small dataset subset for a loader smoke test before launching LoRA training.

Colab reality check:

- Wan2.1 14B LoRA training is unlikely to be comfortable on free T4.
- A100/L4 with gradient checkpointing/offload is much more realistic.
- Save checkpoints to Google Drive, not to the repo.

## 10. Optional Animation

Animation uses the LHM workflow under `4-animation`.

First run SMPL prediction only:

```bash
cd /content/HumanLift_Repro/4-animation
bash predict.sh
```

Then update `inference.sh`:

- `IMAGE_INPUT`: reference/T-pose image
- `MOTION_SEQS_DIR`: SMPL motion folder
- `DATASET_DIR`: multi-view RGBA image folder

Run:

```bash
bash inference.sh
```

## 11. Experiment Log Template

Create one log per run outside large output folders:

```text
date:
git commit:
colab gpu:
torch/cuda:
input image:
preprocess command:
rgb checkpoint:
normal checkpoint:
rgb output folder:
normal output folder:
background removal method:
3gs command:
3gs output folder:
qualitative result:
known issues:
```

Commit small logs if they help reproduce results. Do not commit generated images unless they are deliberately selected small figures for a report.

## 12. First Milestone Checklist

- [ ] `origin` points to your GitHub repo.
- [ ] `upstream` points to the official repo.
- [ ] Initial code baseline pushed to `origin/main`.
- [ ] Colab can clone/pull your repo.
- [ ] One image passes preprocess.
- [ ] RGB multi-view generation produces 81 frames.
- [ ] Normal generation produces 81 frames.
- [ ] Background-removed RGBA images are named `lgt0_r_0000.png` to `lgt0_r_0080.png`.
- [ ] 3DGS smoke test reaches 200 iterations.
- [ ] Full 3DGS run saves a usable output.
- [ ] All commands and checkpoint versions are recorded.
