import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from scene_reconstruction.camera import PluckerEncoder
from scene_reconstruction.gaussian_head import GaussianHead  # your existing grid head
from scene_reconstruction.plucker import plucker_embedding


# -------------------------
# Helpers
# -------------------------

def _safe_normalize_quat(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # q: [..., 4]
    return q / (q.norm(dim=-1, keepdim=True).clamp(min=eps))


# -------------------------
# Patchify / Unpatchify
# -------------------------

class Patchify2x2(nn.Module):
    def __init__(self, in_ch, hidden):
        super().__init__()
        self.proj = nn.Linear(in_ch * 2 * 2, hidden)

    def forward(self, x):
        """
        x: [B, C, h, w]
        -> tokens: [B, N, hidden], where N=(h/2)*(w/2)
        """
        B, C, h, w = x.shape
        assert h % 2 == 0 and w % 2 == 0
        patches = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        return self.proj(patches), (h // 2, w // 2)


class Unpatchify2x2(nn.Module):
    def __init__(self, hidden, out_ch):
        super().__init__()
        self.proj = nn.Linear(hidden, out_ch * 2 * 2)

    def forward(self, tokens, hw2):
        """
        tokens: [B, N, hidden]
        hw2: (h2, w2)
        -> [B, out_ch, h, w]
        """
        h2, w2 = hw2
        x = self.proj(tokens)
        x = rearrange(
            x, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=h2, w=w2, ph=2, pw=2
        )
        return x


# -------------------------
# Transformer backbone
# -------------------------

class ReconNet(nn.Module):
    def __init__(self, hidden=256, layers=8, heads=8, mlp=1024):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=mlp,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.net = nn.TransformerEncoder(enc_layer, num_layers=layers)

    def forward(self, tokens):
        return self.net(tokens)


# -------------------------
# True 3D Gaussian head (points / N gaussians)
# -------------------------

class PointGaussianHead(nn.Module):
    """
    Takes a feature map [B, hidden, h, w] and outputs N gaussians:
      pos:     [B, N, 3]
      scale:   [B, N, 3]   (positive)
      quat:    [B, N, 4]   (normalized)
      opacity: [B, N, 1]   (0..1)
      rgb:     [B, N, 3]   (0..1)

    N defaults to h*w (one gaussian per latent pixel), or you can set num_points.
    """
    def __init__(self, hidden: int, num_points: int | None = None):
        super().__init__()
        self.num_points = num_points  # if None -> use h*w

        # Map per-point features -> gaussian params (14 dims)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 14),
        )

        # Optional: if you want fewer gaussians than h*w, we add a scorer for top-k selection
        self.score = nn.Linear(hidden, 1)

    def forward(self, feat: torch.Tensor):
        """
        feat: [B, hidden, h, w]
        """
        B, Hc, h, w = feat.shape
        tokens = rearrange(feat, "b c h w -> b (h w) c")  # [B, hw, hidden]

        hw = tokens.shape[1]
        N = self.num_points if self.num_points is not None else hw
        if N > hw:
            raise ValueError(f"num_points={N} > h*w={hw}. Reduce num_points or increase feature resolution.")

        if N < hw:
            # top-k selection (cheap + deterministic)
            scores = self.score(tokens).squeeze(-1)  # [B, hw]
            idx = scores.topk(k=N, dim=1, largest=True).indices  # [B, N]
            tokens = tokens.gather(1, idx.unsqueeze(-1).expand(-1, -1, Hc))  # [B, N, hidden]

        out = self.mlp(tokens)  # [B, N, 14]

        pos = out[..., 0:3]

        # Use exp for positive scales, clamp to avoid exploding
        scale = torch.exp(out[..., 3:6].clamp(min=-8.0, max=4.0))

        quat = _safe_normalize_quat(out[..., 6:10])

        opacity = torch.sigmoid(out[..., 10:11])

        rgb = torch.sigmoid(out[..., 11:14])

        return {
            "pos": pos,
            "scale": scale,
            "quat": quat,
            "opacity": opacity,
            "rgb": rgb,
        }


# -------------------------
# Main model
# -------------------------

class MiniLyraStudent(nn.Module):
    """
    head_mode:
      - "grid"   -> your existing GaussianHead output compatible with ImageProxyRenderer
                   (pos/rgb/opacity on a grid)
      - "points" -> true 3D gaussian set: (pos/scale/quat/opacity/rgb) as N points
                   (for real 3DGS renderer backend)
    """

    def __init__(
        self,
        z_ch: int = 32,
        e_ch: int = 32,
        hidden: int = 256,
        layers: int = 8,
        heads: int = 8,
        head_mode: str = "points",          # "grid" or "points"
        num_points: int | None = None,    # only for points head
        plucker_hw_mode: str = "rgb",     # "rgb" or "latent"
        latent_downscale: int = 8,        # if plucker_hw_mode="rgb": H=h*latent_downscale
    ):
        super().__init__()
        assert head_mode in ("grid", "points")
        assert plucker_hw_mode in ("rgb", "latent")

        self.head_mode = head_mode
        self.plucker_hw_mode = plucker_hw_mode
        self.latent_downscale = latent_downscale

        self.plucker_enc = PluckerEncoder(in_ch=6, out_ch=e_ch)
        self.patch_z = Patchify2x2(in_ch=z_ch, hidden=hidden)
        self.patch_e = Patchify2x2(in_ch=e_ch, hidden=hidden)

        self.recon = ReconNet(hidden=hidden, layers=layers, heads=heads, mlp=hidden * 4)
        self.unpatch = Unpatchify2x2(hidden=hidden, out_ch=hidden)

        if self.head_mode == "grid":
            self.head = GaussianHead(hidden=hidden)  # your existing grid head
        else:
            self.head = PointGaussianHead(hidden=hidden, num_points=num_points)

    def _build_plucker_latents(self, Ks, Rs, ts, h, w, device):
        """
        Builds Plücker embeddings per (B,V,L) and encodes them to latent resolution.
        Returns E_lat: [B,V,L,e_ch,h,w]
        """
        B, V, L = Ks.shape[:3]

        if self.plucker_hw_mode == "latent":
            H, W = h, w
        else:
            H, W = h * self.latent_downscale, w * self.latent_downscale

        E_list = []
        for b in range(B):
            for v in range(V):
                for t in range(L):
                    E_hw6 = plucker_embedding(Ks[b, v, t], Rs[b, v, t], ts[b, v, t], H, W)  # [H,W,6]
                    E_list.append(E_hw6.permute(2, 0, 1))  # [6,H,W]

        E = torch.stack(E_list, dim=0).to(device)  # [B*V*L,6,H,W]
        E_lat = self.plucker_enc(E)                # expected to become [B*V*L,e_ch,h,w] (downsample inside encoder)

        # If your PluckerEncoder does NOT downsample to (h,w), enforce it:
        if E_lat.shape[-2:] != (h, w):
            E_lat = F.interpolate(E_lat, size=(h, w), mode="bilinear", align_corners=False)

        E_lat = E_lat.view(B, V, L, E_lat.shape[1], h, w)
        return E_lat

    def forward(self, Z, Ks, Rs, ts):
        """
        Z:  [B,V,L,C,h,w]
        Ks: [B,V,L,3,3]
        Rs: [B,V,L,3,3]
        ts: [B,V,L,3]

        Returns:
          if head_mode="grid":
            dict with values shaped like [B,V,L,*,h,w] (whatever GaussianHead returns)
          if head_mode="points":
            dict with values shaped [B,V,L,N,*]  (pos/scale/quat/opacity/rgb)
        """
        B, V, L, C, h, w = Z.shape
        device = Z.device

        # Plücker latents
        E_lat = self._build_plucker_latents(Ks, Rs, ts, h, w, device)  # [B,V,L,e_ch,h,w]

        # Flatten BV L into batch for patchify
        Z2 = Z.view(B * V * L, C, h, w)
        E2 = E_lat.view(B * V * L, E_lat.shape[3], h, w)

        tz, hw2 = self.patch_z(Z2)  # [BVL, N, hidden]
        te, _ = self.patch_e(E2)    # [BVL, N, hidden]
        tokens = tz + te

        # Multi-view/time fusion:
        # [BVL, N, hidden] -> [B, VL*N, hidden]
        tokens = tokens.view(B, V * L, tokens.shape[1], tokens.shape[2])      # [B, VL, N, hidden]
        tokens = tokens.reshape(B, (V * L) * tokens.shape[2], tokens.shape[3])  # [B, VL*N, hidden]

        tokens = self.recon(tokens)  # [B, VL*N, hidden]

        # Back to per-(B,V,L)
        tokens = tokens.view(B, V * L, -1, tokens.shape[-1])        # [B, VL, N, hidden]
        tokens = tokens.view(B * V * L, -1, tokens.shape[-1])       # [BVL, N, hidden]

        # Feature map
        feat = self.unpatch(tokens, hw2)  # [BVL, hidden, h, w]

        # Head
        out = self.head(feat)

        # Reshape outputs
        if self.head_mode == "grid":
            # GaussianHead returns dict of [BVL, ..., h, w]
            final = {}
            for k, vv in out.items():
                final[k] = vv.view(B, V, L, *vv.shape[1:])
            return final

        else:
            # Point head returns dict of [BVL, N, ...]
            final = {}
            for k, vv in out.items():
                final[k] = vv.view(B, V, L, *vv.shape[1:])
            return final
