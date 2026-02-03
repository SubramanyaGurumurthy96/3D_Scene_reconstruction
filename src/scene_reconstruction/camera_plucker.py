import torch

def pixel_grid(H, W, device):
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )
    return xs.float(), ys.float()

def plucker_from_camera(K, R, t, H, W):
    """
    Assumes camera-to-world transform: X_world = R @ X_cam + t
    If your dataset gives world-to-camera, invert once in loader and keep consistent.
    """
    device = K.device
    xs, ys = pixel_grid(H, W, device)
    ones = torch.ones_like(xs)
    pix = torch.stack([xs, ys, ones], dim=-1)      # [H,W,3]

    Kinv = torch.inverse(K)
    dirs_cam = pix @ Kinv.T                        # [H,W,3]
    dirs_cam = dirs_cam / (torch.norm(dirs_cam, dim=-1, keepdim=True) + 1e-8)

    dirs_world = dirs_cam @ R.T                    # [H,W,3]
    origin = t.view(1, 1, 3).expand(H, W, 3)       # [H,W,3]

    moment = torch.cross(origin, dirs_world, dim=-1)
    E = torch.cat([dirs_world, moment], dim=-1)    # [H,W,6]
    return E
