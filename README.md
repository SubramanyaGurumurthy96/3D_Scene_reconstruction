# 3D Scene Reconstruction (MiniLyra)

This repo trains a MiniLyra student model for 3D scene reconstruction from the Lyra demo dataset format.

## Package layout

- Python package: `scene_reconstruction`
- Training entrypoint: `src/scene_reconstruction/train_minilyra.py`
- Convenience runner: `train_minilyra.py` (repo root)

## Quick start

From the repo root:

```bash
python train_minilyra.py --data-root /workspace/assets/demo --no-depth
```

This auto-detects the demo Lyra layout and uses `static/diffusion_output/0` by default.

## Dataset formats

### 1) Lyra demo format (supported)

Expected structure:

```
assets/demo/
  static/
    diffusion_output/
      0/
        rgb/*.mp4
        pose/*.npz
        intrinsics/*.npz
        latent/*.pkl
        depth/*.zip   (optional, OpenEXR)
```

Run:

```bash
python train_minilyra.py --data-root /workspace/assets/demo --no-depth
```

To use dynamic sequences:

```bash
python train_minilyra.py --data-root /workspace/assets/demo/dynamic/diffusion_output/0 --no-depth
```

### 2) Teacher subset format (optional)

If you have a custom dataset with `train/`, `val/`, `test/` splits, see `src/scene_reconstruction/dataset.py` for the expected layout.

## Training options

- `--epochs N` set number of epochs (default 1)
- `--save-best` save `ckpt_best.pt` when loss improves
- `--no-depth` skip depth loss (recommended if OpenEXR is unavailable)
- `--lpips` enable LPIPS (uses more GPU memory)

Example:

```bash
python train_minilyra.py \
  --data-root /workspace/assets/demo \
  --epochs 100 \
  --save-best \
  --no-depth
```

## Outputs

Checkpoints are written to:

```
/workspace/outputs/minilyra/
  ckpt_best.pt
  ckpt_step_*.pt
  ckpt_final.pt
```

## Notes

- The current renderer is a proxy (image-based) for training stability. It is **not** a full 3D Gaussian rasterizer.
- For real 3DGS rendering, implement `GaussianRenderer` in `src/scene_reconstruction/renderer.py`.

## Install (optional)

```bash
pip install -e .
train-minilyra --data-root /workspace/assets/demo --no-depth
```
