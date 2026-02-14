import torch
import gsplat

class Gaussian2DRenderer:
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


class GaussianRenderer:
    def __init__(self):
        pass

    def render(self, gauss, K, R, t, H, W):
        """
        gauss:
            pos      [N,3]
            scale    [N,3]
            quat     [N,4]
            opacity  [N,1]
            rgb      [N,3]
        """

        means3D = gauss["pos"]              # [N,3]
        scales = gauss["scale"]             # [N,3]
        rotations = gauss["quat"]           # [N,4]
        opacity = gauss["opacity"]          # [N,1]
        colors = gauss["rgb"]               # [N,3]

        # Camera matrices
        viewmat = torch.eye(4, device=means3D.device)
        viewmat[:3,:3] = R
        viewmat[:3,3] = t

        projmat = torch.zeros((4,4), device=means3D.device)
        projmat[0,0] = 2*K[0,0]/W
        projmat[1,1] = 2*K[1,1]/H
        projmat[0,2] = 1 - 2*K[0,2]/W
        projmat[1,2] = 2*K[1,2]/H - 1
        projmat[2,2] = 1
        projmat[3,2] = 1

        render_pkg = gsplat.rasterization(
            means3D,
            scales,
            rotations,
            opacity,
            colors,
            viewmat,
            projmat,
            H,
            W,
        )

        rgb = render_pkg["rgb"]         # [H,W,3]
        depth = render_pkg["depth"]     # [H,W]

        rgb = rgb.permute(2,0,1)
        depth = depth.unsqueeze(0)

        return rgb, depth
