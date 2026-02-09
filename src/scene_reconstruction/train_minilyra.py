import argparse
import os
import time
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from scene_reconstruction.dataset import LyraTeacherSubset, LyraDiffusionOutputDataset
from scene_reconstruction.model import MiniLyraStudent
from scene_reconstruction.losses import compute_losses
from scene_reconstruction.renderer import GaussianRenderer


@dataclass
class TrainState:
    epoch: int
    step: int
    best_loss: float


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_renderer(backend: str):
    if backend == "dummy":
        return DummyRenderer()
    if backend == "image":
        from scene_reconstruction.renderer import ImageProxyRenderer
        return ImageProxyRenderer()
    return GaussianRenderer(backend)


class DummyRenderer:
    """
    Debug-only renderer so the loop runs without a 3DGS backend.
    Returns zeros with the right shape.
    """

    def render(self, gauss, K, R, t, H, W):
        device = gauss["pos"].device
        rgb = torch.zeros(3, H, W, device=device)
        depth = torch.zeros(1, H, W, device=device)
        return rgb, depth


def save_ckpt(path, model, optim, scaler, state: TrainState):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": state.epoch,
            "step": state.step,
            "best_loss": state.best_loss,
        },
        path,
    )


def load_ckpt(path, model, optim, scaler):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    optim.load_state_dict(ckpt["optim"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return TrainState(
        epoch=ckpt.get("epoch", 0),
        step=ckpt.get("step", 0),
        best_loss=ckpt.get("best_loss", float("inf")),
    )


def train_one_epoch(model, loader, renderer, optim, scaler, device, args, state: TrainState):
    model.train()
    accum = max(1, args.grad_accum)
    running_loss = 0.0
    running_steps = 0

    for batch in loader:
        z = batch["z"]
        if z is None:
            raise RuntimeError(
                "Missing latents. Provide latents.npz with key 'z' or add an encoder."
            )

        z = z.to(device)  # [B,V,L,C,h,w]
        Ks = batch["K"].to(device)
        Rs = batch["R"].to(device)
        ts = batch["t"].to(device)
        teacher_rgb = batch["rgb"].to(device)
        teacher_depth = batch["depth"].to(device)
        depth_valid = batch.get("depth_valid", True)
        if torch.is_tensor(depth_valid):
            if depth_valid.numel() == 1:
                depth_valid = bool(depth_valid.item())
            else:
                depth_valid = bool(depth_valid.all().item())
        else:
            depth_valid = bool(depth_valid)

        B, V, L = z.shape[:3]
        H = teacher_rgb.shape[-2]
        W = teacher_rgb.shape[-1]

        if state.step % accum == 0:
            optim.zero_grad(set_to_none=True)


        if state.step == 0:
            print("depth_valid:", depth_valid)
            print("args.no_depth:", args.no_depth)


        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=args.amp):
            gauss = model(z, Ks, Rs, ts)

            loss_total = 0.0
            logs = None
            for b in range(B):
                for v in range(V):
                    for t in range(L):
                        rgb_pred, depth_pred = renderer.render(
                            {k: gauss[k][b, v, t] for k in gauss},
                            Ks[b, v, t],
                            Rs[b, v, t],
                            ts[b, v, t],
                            H,
                            W,
                        )
                        loss, logs = compute_losses(
                            rgb_pred,
                            teacher_rgb[b, v, t],
                            depth_pred,
                            teacher_depth[b, v, t] if depth_valid else None,
                            gauss["opacity"][b, v, t],
                            use_lpips=args.use_lpips,
                            # use_depth=depth_valid and (not args.no_depth),
                            use_depth=True,
                        )
                        loss_total = loss_total + loss

            loss_total = loss_total / max(1, B * V * L)
            loss_total = loss_total / accum

        if args.amp:
            scaler.scale(loss_total).backward()
            if (state.step + 1) % accum == 0:
                scaler.step(optim)
                scaler.update()
        else:
            loss_total.backward()
            if (state.step + 1) % accum == 0:
                optim.step()

        if state.step % args.log_every == 0:
            msg = f"step {state.step} | loss {loss_total.item():.4f}"
            if logs:
                msg += " | " + " ".join([f"{k}:{v:.4f}" for k, v in logs.items()])
            print(msg)

        running_loss += float(loss_total.item())
        running_steps += 1

        # Save best checkpoint on improvement
        if args.save_best and loss_total.item() < state.best_loss:
            state.best_loss = float(loss_total.item())
            save_ckpt(
                os.path.join(args.out_dir, "ckpt_best.pt"),
                model,
                optim,
                scaler,
                state,
            )

        if state.step % args.ckpt_every == 0 and state.step > 0:
            save_ckpt(
                os.path.join(args.out_dir, f"ckpt_step_{state.step}.pt"),
                model,
                optim,
                scaler,
                state,
            )

        state.step += 1
    return running_loss / max(1, running_steps)


def parse_args():
    p = argparse.ArgumentParser("MiniLyra training")
    p.add_argument("--data-root", required=True, help="Dataset root")
    p.add_argument("--out-dir", default="/workspace/outputs/minilyra")
    p.add_argument("--split", default="train")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.set_defaults(amp=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--resume", default="")
    p.add_argument("--save-best", action="store_true", help="Save ckpt_best.pt on loss improvement")

    # data layout
    p.add_argument("--views", type=int, default=1, help="V")
    p.add_argument("--frames", type=int, default=0, help="L (0=auto)")
    p.add_argument("--height", type=int, default=0, help="H (0=auto)")
    p.add_argument("--width", type=int, default=0, help="W (0=auto)")

    # model
    p.add_argument("--z-ch", type=int, default=32)
    p.add_argument("--e-ch", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)

    # renderer
    p.add_argument(
        "--renderer-backend",
        default="image",
        help="Backend identifier ('image' proxy, 'dummy' for debug, or real backend)",
    )
    p.add_argument("--data-format", default="auto", choices=["auto", "teacher", "demo"])
    p.add_argument("--no-depth", action="store_true")
    p.add_argument("--lpips", dest="use_lpips", action="store_true", default=False)

    return p.parse_args()


def main():
    args = parse_args()
    seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.data_format == "auto":
        if os.path.isdir(os.path.join(args.data_root, "diffusion_output")):
            data_root = os.path.join(args.data_root, "diffusion_output", "0")
            args.data_format = "demo"
        elif os.path.isdir(os.path.join(args.data_root, "static", "diffusion_output")):
            data_root = os.path.join(args.data_root, "static", "diffusion_output", "0")
            args.data_format = "demo"
        else:
            data_root = args.data_root
            args.data_format = "teacher"
    else:
        data_root = args.data_root

    if args.data_format == "demo":
        ds = LyraDiffusionOutputDataset(
            root=data_root,
            V=args.views,
            L=args.frames,
            H=args.height,
            W=args.width,
            load_latents=True,
            load_depth=not args.no_depth,
        )
    else:
        ds = LyraTeacherSubset(
        root=data_root,
        split=args.split,
        V=args.views,
        L=args.frames if args.frames > 0 else 1,
        H=args.height if args.height > 0 else 176,
        W=args.width if args.width > 0 else 320,
        load_latents=True,
    )
    # Infer z_ch from the dataset if possible
    try:
        sample = ds[0]
        if sample.get("z") is not None:
            inferred_z = int(sample["z"].shape[2])
            if inferred_z != args.z_ch:
                print(f"[info] overriding z_ch {args.z_ch} -> {inferred_z} based on dataset")
                args.z_ch = inferred_z
    except Exception:
        pass

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = MiniLyraStudent(z_ch=args.z_ch, e_ch=args.e_ch, hidden=args.hidden)
    model = model.to(device)

    renderer = build_renderer(args.renderer_backend)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = torch.amp.GradScaler(device_type, enabled=args.amp)

    state = TrainState(epoch=0, step=0, best_loss=float("inf"))
    if args.resume:
        state = load_ckpt(args.resume, model, optim, scaler)

    print(f"Training on {device} for {args.epochs} epoch(s)")
    start = time.time()

    for epoch in range(state.epoch, args.epochs):
        state.epoch = epoch
        avg_loss = train_one_epoch(model, dl, renderer, optim, scaler, device, args, state)
        print(f"epoch {epoch} | avg_loss {avg_loss:.4f} | best {state.best_loss:.4f}")

    elapsed = time.time() - start
    save_ckpt(os.path.join(args.out_dir, "ckpt_final.pt"), model, optim, scaler, state)
    print(f"Done in {elapsed/60.0:.1f} min")


if __name__ == "__main__":
    main()
