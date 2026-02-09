import torch


class GaussianRenderer:
    def __init__(self, backend):
        self.backend = backend  # gsplat / diff-gaussian-rasterization

    def render(self, gauss, K, R, t, H, W):
        """
        gauss: dict pos/scale/quat/opacity/rgb shaped [N, ...] or [H,W,...] depending on backend
        Return: rgb [3,H,W], depth [1,H,W]
        """
        raise NotImplementedError(
            "GaussianRenderer is a stub. Use --renderer-backend image for a "
            "simple training proxy, or plug in a real 3DGS backend."
        )


class ImageProxyRenderer:
    def render(self, gauss, K, R, t, H, W):
        rgb = gauss["rgb"]
        pos = gauss["pos"]

        if rgb.dim() == 4: rgb = rgb.squeeze(0)
        if pos.dim() == 4: pos = pos.squeeze(0)

        rgb_out = rgb.clamp(0.0, 1.0)       # <— no "* opacity"
        depth_out = pos[2:3]
        return rgb_out, depth_out
