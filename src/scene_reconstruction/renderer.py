import torch
import gsplat
import inspect


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
        Robust gsplat renderer that adapts to the installed gsplat.rasterization() signature.
        - Avoids hardcoding argument names like means3D that differ across gsplat versions.
        """

        if not isinstance(gauss, dict):
            raise TypeError(f"Expected gauss dict, got {type(gauss)}")

        def pick_key(d, candidates):
            for k in candidates:
                if k in d:
                    return k
            return None

        # ---- map gaussian keys (aliases) ----
        k_means = pick_key(gauss, ["xyz", "means3D", "means", "pos", "positions", "center", "centers"])
        k_scales = pick_key(gauss, ["scales", "scale", "sigmas", "sigma"])
        k_rots = pick_key(gauss, ["rot", "rots", "rotation", "rotations", "quat", "quats", "q"])
        k_cols = pick_key(gauss, ["rgb", "color", "colors", "f_dc"])
        k_opac = pick_key(gauss, ["opacity", "opacities", "alpha", "alphas", "a"])

        if None in [k_means, k_scales, k_rots, k_cols, k_opac]:
            raise KeyError(
                "GaussianRenderer.render(): couldn't map required gaussian keys.\n"
                f"Available keys: {sorted(list(gauss.keys()))}\n"
                f"Need keys for means/scales/rots/colors/opacity."
            )

        means = gauss[k_means]
        scales = gauss[k_scales]
        quats = gauss[k_rots]
        colors = gauss[k_cols]
        opacities = gauss[k_opac]

        device = means.device
        dtype = means.dtype

        # ---- ensure camera dims ----
        if K.dim() == 2: K = K.unsqueeze(0)  # [1,3,3]
        if R.dim() == 2: R = R.unsqueeze(0)  # [1,3,3]
        if t.dim() == 1: t = t.unsqueeze(0)  # [1,3]

        K = K.to(device=device, dtype=dtype)
        R = R.to(device=device, dtype=dtype)
        t = t.to(device=device, dtype=dtype)

        viewmats = torch.eye(4, device=device, dtype=dtype).unsqueeze(0)  # [1,4,4]
        viewmats[:, :3, :3] = R
        viewmats[:, :3, 3] = t

        # ---- sanitize shapes ----
        means = means.reshape(-1, 3).to(device=device, dtype=dtype)

        scales = scales.to(device=device, dtype=dtype)
        if scales.dim() == 1:
            scales = scales[:, None].repeat(1, 3)
        elif scales.shape[-1] == 1:
            scales = scales.repeat(1, 3)
        scales = scales.reshape(-1, 3)

        quats = quats.reshape(-1, 4).to(device=device, dtype=dtype)

        colors = colors.to(device=device, dtype=dtype)
        colors = colors.reshape(-1, colors.shape[-1])
        if colors.shape[-1] != 3:
            raise ValueError(
                f"Expected colors to be [N,3], got {tuple(colors.shape)} from key '{k_cols}'. "
                "If this is SH/features, you must convert to RGB first."
            )

        opacities = opacities.to(device=device, dtype=dtype).view(-1)

        # ---- build kwargs based on installed gsplat signature ----
        sig = inspect.signature(gsplat.rasterization)
        accepted = set(sig.parameters.keys())

        # Candidate name mapping across gsplat versions
        candidates = {
            # gaussians
            "means3D": means,
            "means3d": means,
            "means": means,
            "xyz": means,

            "scales": scales,
            "scale": scales,

            "rotations": quats,
            "rotation": quats,
            "quats": quats,
            "quat": quats,

            "opacities": opacities,
            "opacity": opacities,
            "alphas": opacities,
            "alpha": opacities,

            "colors": colors,
            "rgb": colors,

            # camera
            "viewmats": viewmats,
            "view_mats": viewmats,
            "views": viewmats,

            "Ks": K,
            "K": K,

            # image size
            "width": int(W),
            "W": int(W),
            "height": int(H),
            "H": int(H),

            # optional flags
            "packed": False,
        }

        kwargs = {}
        for name, val in candidates.items():
            if name in accepted:
                kwargs[name] = val

        # Sanity: must have some required params present
        must_have_any = [
            ["means", "means3D", "means3d", "xyz"],
            ["viewmats", "view_mats", "views"],
            ["Ks", "K"],
        ]
        for group in must_have_any:
            if not any(g in kwargs for g in group if g in accepted):
                raise RuntimeError(
                    "gsplat.rasterization signature mismatch: missing required argument group.\n"
                    f"Accepted args: {sorted(list(accepted))}\n"
                    f"Provided kwargs: {sorted(list(kwargs.keys()))}\n"
                    "Fix: paste the accepted args list; we'll map correctly."
                )

        render_pkg = gsplat.rasterization(**kwargs)

        # Different gsplat versions may return different keys.
        # Common are: "render" and "depth", or tuple outputs.
        if isinstance(render_pkg, dict):
            if "render" in render_pkg:
                rgb_out = render_pkg["render"]
            elif "rgb" in render_pkg:
                rgb_out = render_pkg["rgb"]
            else:
                raise KeyError(f"gsplat output keys: {list(render_pkg.keys())} (no render/rgb)")

            if "depth" in render_pkg:
                depth_out = render_pkg["depth"]
            else:
                depth_out = None
        else:
            # If gsplat returns a tuple, assume (rgb, depth, ...)
            rgb_out = render_pkg[0]
            depth_out = render_pkg[1] if len(render_pkg) > 1 else None

        # rgb_out usually [C,H,W,3] -> [3,H,W]
        if rgb_out.dim() == 4:
            rgb = rgb_out[0].permute(2, 0, 1).contiguous()
        elif rgb_out.dim() == 3:
            # already [H,W,3]
            rgb = rgb_out.permute(2, 0, 1).contiguous()
        else:
            raise RuntimeError(f"Unexpected rgb_out shape: {tuple(rgb_out.shape)}")

        # depth_out usually [C,H,W] -> [1,H,W]
        if depth_out is None:
            depth = None
        else:
            if depth_out.dim() == 3:
                depth = depth_out[0].unsqueeze(0).contiguous()
            elif depth_out.dim() == 2:
                depth = depth_out.unsqueeze(0).contiguous()
            else:
                raise RuntimeError(f"Unexpected depth_out shape: {tuple(depth_out.shape)}")

        return rgb, depth
