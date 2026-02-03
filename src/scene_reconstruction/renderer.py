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
    """
    Minimal differentiable renderer that treats predicted RGB/opacity as image outputs.
    This is NOT a 3DGS rasterizer, but allows training to run end-to-end.
    """

    def render(self, gauss, K, R, t, H, W):
        rgb = gauss["rgb"]
        opacity = gauss["opacity"]
        pos = gauss["pos"]

        # Expect shapes [3,H,W], [1,H,W], [3,H,W]
        if rgb.dim() == 4:
            rgb = rgb.squeeze(0)
        if opacity.dim() == 4:
            opacity = opacity.squeeze(0)
        if pos.dim() == 4:
            pos = pos.squeeze(0)

        # Proxy: alpha-composited RGB and z-depth from predicted position
        rgb_out = (rgb * opacity).clamp(0.0, 1.0)
        depth_out = pos[2:3]
        return rgb_out, depth_out
