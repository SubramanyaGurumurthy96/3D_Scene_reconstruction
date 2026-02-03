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

def compute_losses(render_rgb, teacher_rgb, render_depth, teacher_depth, opacity):
    Lmse = F.mse_loss(render_rgb, teacher_rgb)
    Llp  = _lpips(render_rgb*2-1, teacher_rgb*2-1).mean()
    Ld   = scale_invariant_depth_loss(render_depth, teacher_depth)
    Lop  = opacity.abs().mean()

    loss = 1.0*Lmse + 0.5*Llp + 0.05*Ld + 0.1*Lop  # :contentReference[oaicite:29]{index=29}
    return loss, {"mse":Lmse.item(), "lpips":Llp.item(), "depth":Ld.item(), "opacity":Lop.item()}

def prune_by_opacity(gauss, keep_ratio=0.2):
    # keep top 20% opacity (prune lowest 80%) :contentReference[oaicite:30]{index=30}
    op = gauss["opacity"].reshape(-1)
    k = max(1, int(op.numel()*keep_ratio))
    thresh = torch.topk(op, k).values.min()
    mask = (gauss["opacity"] >= thresh)
    return mask
