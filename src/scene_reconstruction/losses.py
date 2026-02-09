import torch
import torch.nn.functional as F
import lpips

_lpips = lpips.LPIPS(net="vgg").eval()

def scale_invariant_depth_loss(pred, gt, mask=None, eps=1e-6):
    # simple version of scale-invariant loss (log space)
    pred = pred.clamp_min(eps)
    gt   = gt.clamp_min(eps)
    d = torch.log(pred) - torch.log(gt)
    if mask is not None:
        d = d[mask]
    return d.pow(2).mean() - d.mean().pow(2)

def compute_losses(render_rgb, teacher_rgb, render_depth, teacher_depth, opacity,
                   use_lpips=False, use_depth=True):
    # Align teacher resolution to render resolution if needed
    if teacher_rgb.shape != render_rgb.shape:
        import torch.nn.functional as F
        tr = teacher_rgb.unsqueeze(0)
        tr = F.interpolate(tr, size=render_rgb.shape[-2:], mode="bilinear", align_corners=False)
        teacher_rgb = tr.squeeze(0)
    if teacher_depth is not None and teacher_depth.shape != render_depth.shape:
        import torch.nn.functional as F
        td = teacher_depth.unsqueeze(0)
        td = F.interpolate(td, size=render_depth.shape[-2:], mode="nearest")
        teacher_depth = td.squeeze(0)

    Lmse = F.mse_loss(render_rgb, teacher_rgb)
    Llp = torch.tensor(0.0, device=render_rgb.device)
    if use_lpips:
        lpips_model = _lpips.to(render_rgb.device)
        Llp = lpips_model(render_rgb * 2 - 1, teacher_rgb * 2 - 1).mean()

    Ld = torch.tensor(0.0, device=render_rgb.device)
    if use_depth and teacher_depth is not None:
        Ld = scale_invariant_depth_loss(render_depth, teacher_depth)

    Lop = (opacity - 1.0).abs().mean()

    loss = Lmse + (0.5 * Llp if use_lpips else 0.0) + (0.05 * Ld if use_depth else 0.0)

    return loss, {
        "mse": Lmse.item(),
        "lpips": float(Llp.item()),
        "depth": float(Ld.item()),
        "opacity": Lop.item(),
    }

def prune_by_opacity(gauss, keep_ratio=0.2):
    # keep top 20% opacity (prune lowest 80%) :contentReference[oaicite:30]{index=30}
    op = gauss["opacity"].reshape(-1)
    k = max(1, int(op.numel()*keep_ratio))
    thresh = torch.topk(op, k).values.min()
    mask = (gauss["opacity"] >= thresh)
    return mask
