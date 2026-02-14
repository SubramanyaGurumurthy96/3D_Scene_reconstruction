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
        Robust gsplat renderer with flexible key mapping.
        """
        # -------------------------
        # Safety check
        # -------------------------
        if not isinstance(gauss, dict):
            raise TypeError(f"Expected gauss dict, got {type(gauss)}")

        def pick_key(d, candidates):
            for k in candidates:
                if k in d:
                    return k
            return None

        # -------------------------
        # Map possible key names
        # -------------------------
        k_means = pick_key(gauss, ["xyz", "means3D", "means", "pos", "positions"])
        k_scales = pick_key(gauss, ["scales", "scale"])
        k_rots = pick_key(gauss, ["rot", "rotation", "rotations", "quat", "quats"])
        k_cols = pick_key(gauss, ["rgb", "color", "colors"])
        k_opac = pick_key(gauss, ["opacity", "opacities", "alpha"])

        if None in [k_means, k_scales, k_rots, k_cols, k_opac]:
            raise KeyError(
                f"Renderer could not map gaussian keys.\n"
                f"Available keys: {list(gauss.keys())}"
            )

        means3D = gauss[k_means]
        scales = gauss[k_scales]
        rotations = gauss[k_rots]
        colors = gauss[k_cols]
        opacity = gauss[k_opac]

        device = means3D.device
        dtype = means3D.dtype

        # -------------------------
        # Camera dims fix
        # -------------------------
        if K.dim() == 2:
            K = K.unsqueeze(0)
        if R.dim() == 2:
            R = R.unsqueeze(0)
        if t.dim() == 1:
            t = t.unsqueeze(0)

        K = K.to(device=device, dtype=dtype)
        R = R.to(device=device, dtype=dtype)
        t = t.to(device=device, dtype=dtype)

        viewmats = torch.eye(4, device=device, dtype=dtype).unsqueeze(0)
        viewmats[:, :3, :3] = R
        viewmats[:, :3, 3] = t

        # -------------------------
        # Shape sanitation
        # -------------------------
        means3D = means3D.reshape(-1, 3)

        if scales.dim() == 1:
            scales = scales[:, None].repeat(1, 3)
        elif scales.shape[-1] == 1:
            scales = scales.repeat(1, 3)
        scales = scales.reshape(-1, 3)

        rotations = rotations.reshape(-1, 4)

        colors = colors.reshape(-1, 3)
        opacity = opacity.view(-1)

        # -------------------------
        # Rasterize
        # -------------------------
        render_pkg = gsplat.rasterization(
            means3D=means3D,
            scales=scales,
            rotations=rotations,
            opacities=opacity,
            colors=colors,
            viewmats=viewmats,
            Ks=K,
            width=W,
            height=H,
            packed=False,
        )

        rgb = render_pkg["render"][0].permute(2, 0, 1)  # [3,H,W]
        depth = render_pkg["depth"][0].unsqueeze(0)     # [1,H,W]

        return rgb, depth
