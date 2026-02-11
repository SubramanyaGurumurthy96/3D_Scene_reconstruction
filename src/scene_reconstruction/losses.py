import torch
import torch.nn.functional as F

_lpips = None  # lazy init


def scale_invariant_depth_loss(pred, gt, mask=None, eps=1e-6):
    pred = pred.clamp_min(eps)
    gt   = gt.clamp_min(eps)
    d = torch.log(pred) - torch.log(gt)
    if mask is not None:
        d = d[mask]
    return d.pow(2).mean() - d.mean().pow(2)


def compute_losses(
    render_rgb,
    teacher_rgb,
    render_depth,
    teacher_depth,
    opacity,
    use_lpips=False,
    use_depth=True,
):
    global _lpips

    print("use_depth flag:", use_depth)

    print("render_depth mean:", render_depth.mean().item())
    print("teacher_depth mean:", teacher_depth.mean().item())


    # Resize teacher to match render resolution
    if teacher_rgb.shape != render_rgb.shape:
        teacher_rgb = F.interpolate(
            teacher_rgb.unsqueeze(0),
            size=render_rgb.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    if teacher_depth is not None and teacher_depth.shape != render_depth.shape:
        teacher_depth = F.interpolate(
            teacher_depth.unsqueeze(0),
            size=render_depth.shape[-2:],
            mode="nearest",
        ).squeeze(0)

    # RGB loss
    Lmse = F.mse_loss(render_rgb, teacher_rgb)

    # LPIPS (lazy, optional)
    Llp = torch.tensor(0.0, device=render_rgb.device)
    if use_lpips:
        if _lpips is None:
            import lpips
            print("Setting up [LPIPS] perceptual loss")
            _lpips = lpips.LPIPS(net="vgg").to(render_rgb.device).eval()
        Llp = _lpips(render_rgb, teacher_rgb).mean()

    # Depth loss
    Ld = torch.tensor(0.0, device=render_rgb.device)
    if use_depth and teacher_depth is not None:
        Ld = scale_invariant_depth_loss(render_depth, teacher_depth)

    # Total loss (NO opacity penalty in proxy training)
    loss = Lmse
    if use_lpips:
        loss = loss + 0.5 * Llp
    if use_depth:
        loss = loss + 0.05 * Ld

    return loss, {
        "mse": Lmse.item(),
        "lpips": float(Llp.item()),
        "depth": float(Ld.item()),
        "opacity": opacity.mean().item(),
    }


def prune_by_opacity(gauss, keep_ratio=0.2):
    op = gauss["opacity"].reshape(-1)
    k = max(1, int(op.numel() * keep_ratio))
    thresh = torch.topk(op, k).values.min()
    mask = (gauss["opacity"] >= thresh)
    return mask
