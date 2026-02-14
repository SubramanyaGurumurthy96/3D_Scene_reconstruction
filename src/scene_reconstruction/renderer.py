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
        gauss: dict with keys
            xyz      [N,3]
            scales   [N,3]
            rot      [N,4]
            opacity  [N] or [N,1]
            rgb      [N,3]

        K: [3,3]
        R: [3,3]
        t: [3]
        """

        device = gauss["xyz"].device
        dtype = gauss["xyz"].dtype

        # ---------------------------
        # Ensure camera batch dimension
        # ---------------------------

        if K.dim() == 2:
            K = K.unsqueeze(0)  # [1,3,3]

        if R.dim() == 2:
            R = R.unsqueeze(0)  # [1,3,3]

        if t.dim() == 1:
            t = t.unsqueeze(0)  # [1,3]

        K = K.to(device=device, dtype=dtype)
        R = R.to(device=device, dtype=dtype)
        t = t.to(device=device, dtype=dtype)

        # ---------------------------
        # Build world->camera view matrix
        # ---------------------------

        viewmats = torch.eye(4, device=device, dtype=dtype).unsqueeze(0)  # [1,4,4]
        viewmats[:, :3, :3] = R
        viewmats[:, :3, 3] = t

        # ---------------------------
        # Prepare Gaussian parameters
        # ---------------------------

        means3D = gauss["xyz"]                  # [N,3]
        scales = gauss["scales"]                # [N,3]
        rotations = gauss["rot"]                # [N,4] quaternion
        opacity = gauss["opacity"].view(-1)     # [N]
        colors = gauss["rgb"]                   # [N,3]

        # ---------------------------
        # Call gsplat rasterizer
        # ---------------------------

        render_pkg = gsplat.rasterization(
            means3D=means3D,
            scales=scales,
            rotations=rotations,
            opacities=opacity,
            colors=colors,
            viewmats=viewmats,   # [C,4,4]
            Ks=K,                # [C,3,3]
            width=W,
            height=H,
            packed=False,
        )

        # render_pkg contains:
        #   "render" -> [C,H,W,3]
        #   "depth"  -> [C,H,W]

        rgb = render_pkg["render"][0].permute(2, 0, 1)   # [3,H,W]
        depth = render_pkg["depth"][0].unsqueeze(0)      # [1,H,W]

        return rgb, depth
