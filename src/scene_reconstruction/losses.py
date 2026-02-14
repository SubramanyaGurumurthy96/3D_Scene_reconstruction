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


def charbonnier_loss(x, y, eps=1e-6):
    return torch.mean(torch.sqrt((x - y) ** 2 + eps))


def compute_losses(
    render_rgb,
    teacher_rgb,
    render_depth,
    teacher_depth,
    opacity,
    use_lpips=True,
    use_depth=True,
    use_reg=False,          # enable when using real 3D renderer
    gauss_scale=None,       # optional scale regularization
):
    global _lpips

    device = render_rgb.device

    # -------------------------------------------------------
    # Resize teacher to match render resolution
    # -------------------------------------------------------
    if teacher_rgb.shape != render_rgb.shape:
        teacher_rgb = F.interpolate(
            teacher_rgb.unsqueeze(0),
            size=render_rgb.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    if (
        teacher_depth is not None
        and render_depth is not None
        and teacher_depth.shape != render_depth.shape
    ):
        teacher_depth = F.interpolate(
            teacher_depth.unsqueeze(0),
            size=render_depth.shape[-2:],
            mode="nearest",
        ).squeeze(0)

    # -------------------------------------------------------
    # RGB loss (Charbonnier is more stable than MSE)
    # -------------------------------------------------------
    Lrgb = charbonnier_loss(render_rgb, teacher_rgb)

    # -------------------------------------------------------
    # LPIPS (normalize to [-1,1] as required)
    # -------------------------------------------------------
    Llp = torch.tensor(0.0, device=device)
    if use_lpips:
        if _lpips is None:
            import lpips
            print("Setting up [LPIPS] perceptual loss")
            _lpips = lpips.LPIPS(net="vgg").to(device).eval()

        # LPIPS expects [-1,1]
        r1 = render_rgb * 2.0 - 1.0
        r2 = teacher_rgb * 2.0 - 1.0
        Llp = _lpips(r1, r2).mean()

    # -------------------------------------------------------
    # Depth loss
    # -------------------------------------------------------
    Ld = torch.tensor(0.0, device=device)
    if use_depth and teacher_depth is not None:
        Ld = scale_invariant_depth_loss(render_depth, teacher_depth)

    # -------------------------------------------------------
    # Optional regularization (ONLY for real 3D renderer)
    # -------------------------------------------------------
    Lopacity = torch.tensor(0.0, device=device)
    Lscale = torch.tensor(0.0, device=device)

    if use_reg:
        # Encourage sparse opacity
        Lopacity = torch.mean(opacity)

        # Prevent exploding Gaussian size
        if gauss_scale is not None:
            Lscale = torch.mean(torch.abs(gauss_scale))

    # -------------------------------------------------------
    # Final weighted loss
    # -------------------------------------------------------

    loss = (
        Lrgb
        + 0.3 * Llp
        + 0.1 * Ld
    )

    if use_reg:
        loss = loss + 0.01 * Lopacity + 0.001 * Lscale

    return loss, {
        "rgb": Lrgb.item(),
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
