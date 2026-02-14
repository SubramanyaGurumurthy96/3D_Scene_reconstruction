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
        gsplat renderer that:
        - maps gaussian keys robustly
        - adapts to installed gsplat.rasterization() signature
        - FORCES fp32 to avoid: kernel not implemented for Half
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
                f"Available keys: {sorted(list(gauss.keys()))}"
            )
    
        # Use the same device as means; dtype forced to float32 later
        means = gauss[k_means]
        device = means.device
    
        # -------------------------
        # Force fp32 for gsplat (critical)
        # -------------------------
        with torch.cuda.amp.autocast(enabled=False):
            dtype = torch.float32
    
            means = gauss[k_means].reshape(-1, 3).to(device=device, dtype=dtype)
    
            scales = gauss[k_scales].to(device=device, dtype=dtype)
            if scales.dim() == 1:
                scales = scales[:, None].repeat(1, 3)
            elif scales.shape[-1] == 1:
                scales = scales.repeat(1, 3)
            scales = scales.reshape(-1, 3)
    
            quats = gauss[k_rots].reshape(-1, 4).to(device=device, dtype=dtype)
    
            colors = gauss[k_cols].to(device=device, dtype=dtype)
            colors = colors.reshape(-1, colors.shape[-1])
            if colors.shape[-1] != 3:
                raise ValueError(
                    f"Expected colors to be [N,3], got {tuple(colors.shape)} from key '{k_cols}'. "
                    "If this is SH/features, convert to RGB first."
                )
    
            opacities = gauss[k_opac].to(device=device, dtype=dtype).view(-1)
    
            # ---- camera dims ----
            if K.dim() == 2: K = K.unsqueeze(0)
            if R.dim() == 2: R = R.unsqueeze(0)
            if t.dim() == 1: t = t.unsqueeze(0)
    
            K = K.to(device=device, dtype=dtype)
            R = R.to(device=device, dtype=dtype)
            t = t.to(device=device, dtype=dtype)
    
            viewmats = torch.eye(4, device=device, dtype=dtype).unsqueeze(0)  # [1,4,4]
            viewmats[:, :3, :3] = R
            viewmats[:, :3, 3] = t
    
            # ---- build kwargs based on installed gsplat signature ----
            sig = inspect.signature(gsplat.rasterization)
            accepted = set(sig.parameters.keys())
    
            candidates = {
                # gaussians
                "means3D": means, "means3d": means, "means": means, "xyz": means,
                "scales": scales, "scale": scales,
                "rotations": quats, "rotation": quats, "quats": quats, "quat": quats,
                "opacities": opacities, "opacity": opacities, "alphas": opacities, "alpha": opacities,
                "colors": colors, "rgb": colors,
    
                # camera
                "viewmats": viewmats, "view_mats": viewmats, "views": viewmats,
                "Ks": K, "K": K,
    
                # image size
                "width": int(W), "W": int(W),
                "height": int(H), "H": int(H),
    
                # optional flags
                "packed": False,
            }
    
            kwargs = {name: val for name, val in candidates.items() if name in accepted}
    
            render_pkg = gsplat.rasterization(**kwargs)
    
            # ---- unpack outputs ----
            if isinstance(render_pkg, dict):
                rgb_out = render_pkg["render"] if "render" in render_pkg else render_pkg.get("rgb", None)
                depth_out = render_pkg.get("depth", None)
                if rgb_out is None:
                    raise KeyError(f"gsplat output keys: {list(render_pkg.keys())} (no render/rgb)")
            else:
                rgb_out = render_pkg[0]
                depth_out = render_pkg[1] if len(render_pkg) > 1 else None
    
            # rgb_out typically [C,H,W,3]
            if rgb_out.dim() == 4:
                rgb = rgb_out[0].permute(2, 0, 1).contiguous()   # [3,H,W]
            elif rgb_out.dim() == 3:
                rgb = rgb_out.permute(2, 0, 1).contiguous()
            else:
                raise RuntimeError(f"Unexpected rgb_out shape: {tuple(rgb_out.shape)}")
    
            if depth_out is None:
                depth = None
            else:
                # Case 1: [C,H,W,1]
                if depth_out.dim() == 4 and depth_out.shape[-1] == 1:
                    depth = depth_out[0].permute(2, 0, 1).contiguous()  # [1,H,W]

                # Case 2: [C,H,W]
                elif depth_out.dim() == 3:
                    depth = depth_out[0].unsqueeze(0).contiguous()      # [1,H,W]

                # Case 3: [H,W]
                elif depth_out.dim() == 2:
                    depth = depth_out.unsqueeze(0).contiguous()

                else:
                    raise RuntimeError(f"Unexpected depth_out shape: {tuple(depth_out.shape)}")

    
        return rgb, depth
