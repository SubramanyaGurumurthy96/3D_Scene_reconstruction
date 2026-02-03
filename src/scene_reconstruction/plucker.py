import torch

def pixel_grid(H, W, device):
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )
    return xs.float(), ys.float()  # [H,W]

def rays_from_KRt(K, R, t, H, W):
    """
    K: [3,3], R:[3,3], t:[3]  (camera-to-world OR world-to-camera — be consistent!)
    We'll assume camera-to-world here:
      x_world = R @ x_cam + t
    """
    device = K.device
    xs, ys = pixel_grid(H, W, device)
    ones = torch.ones_like(xs)
    pix = torch.stack([xs, ys, ones], dim=-1)  # [H,W,3]

    Kinv = torch.inverse(K)
    dirs_cam = (pix @ Kinv.T)  # [H,W,3] in camera coords
    dirs_cam = dirs_cam / (torch.norm(dirs_cam, dim=-1, keepdim=True) + 1e-8)

    # to world
    dirs_world = dirs_cam @ R.T  # [H,W,3]
    origin_world = t.view(1, 1, 3).expand(H, W, 3)

    return origin_world, dirs_world

def plucker_embedding(K, R, t, H, W):
    o, d = rays_from_KRt(K, R, t, H, W)
    m = torch.cross(o, d, dim=-1)
    E = torch.cat([d, m], dim=-1)  # [H,W,6]
    return E
