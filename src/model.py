import torch
import torch.nn as nn
from einops import rearrange
import torch
import torch.nn as nn
from einops import rearrange
from .camera import PluckerEncoder
from .gaussian_head import GaussianHead
from .plucker import plucker_embedding
from .model import Patchify2x2, Unpatchify2x2  # if split files adjust imports



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
        return self.proj(patches), (h//2, w//2)

class Unpatchify2x2(nn.Module):
    def __init__(self, hidden, out_ch):
        super().__init__()
        self.proj = nn.Linear(hidden, out_ch * 2 * 2)

    def forward(self, tokens, hw2):
        """
        tokens: [B, N, hidden]
        hw2: (h2, w2)
        -> [B,out_ch,h,w]
        """
        h2, w2 = hw2
        B, N, _ = tokens.shape
        x = self.proj(tokens)
        x = rearrange(x, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h2, w=w2, ph=2, pw=2)
        return x

class ReconNet(nn.Module):
    def __init__(self, hidden=256, layers=8, heads=8, mlp=1024):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads,
            dim_feedforward=mlp, batch_first=True, activation="gelu"
        )
        self.net = nn.TransformerEncoder(enc_layer, num_layers=layers)

    def forward(self, tokens):
        return self.net(tokens)

class MiniLyraStudent(nn.Module):
    def __init__(self, z_ch=32, e_ch=32, hidden=256):
        super().__init__()
        self.plucker_enc = PluckerEncoder(in_ch=6, out_ch=e_ch)
        self.patch_z = Patchify2x2(in_ch=z_ch, hidden=hidden)
        self.patch_e = Patchify2x2(in_ch=e_ch, hidden=hidden)

        self.recon = ReconNet(hidden=hidden, layers=8, heads=8, mlp=hidden*4)
        self.unpatch = Unpatchify2x2(hidden=hidden, out_ch=hidden)
        self.head = GaussianHead(hidden=hidden)

    def forward(self, Z, Ks, Rs, ts):
        """
        Z: [B,V,L,C,h,w]  (your latent tensor)
        Ks,Rs,ts: camera params for each (V,L)
          Ks: [B,V,L,3,3], Rs:[B,V,L,3,3], ts:[B,V,L,3]
        Return: Gaussian params per (B,V,L,H,W)
        """
        B,V,L,C,h,w = Z.shape

        # Build Plücker E at RGB resolution H,W (you choose H,W consistent with teacher frames)
        # For a mini version, set H,W = h*8, w*8 (if latent downsample factor is 8).
        H, W = h*8, w*8
        E_list = []
        for b in range(B):
            for v in range(V):
                for t in range(L):
                    E_hw6 = plucker_embedding(Ks[b,v,t], Rs[b,v,t], ts[b,v,t], H, W)  # [H,W,6]
                    E_list.append(E_hw6.permute(2,0,1))  # [6,H,W]
        E = torch.stack(E_list, dim=0)  # [B*V*L,6,H,W]
        E_lat = self.plucker_enc(E)     # [B*V*L,e_ch,h,w] (downsampled)
        E_lat = E_lat.view(B,V,L,-1,E_lat.shape[-2],E_lat.shape[-1])

        # Flatten BV L into batch for patchify
        Z2 = Z.view(B*V*L, C, h, w)
        E2 = E_lat.view(B*V*L, E_lat.shape[3], h, w)

        tz, hw2 = self.patch_z(Z2)
        te, _   = self.patch_e(E2)
        tokens = tz + te  # paper sums inputs before recon blocks :contentReference[oaicite:20]{index=20}

        # Now multi-view fusion:
        # simplest: group by B and let attention mix across all (V*L) by concatenating token sequences
        tokens = tokens.view(B, V*L, tokens.shape[1], tokens.shape[2])  # [B,VL,N,hidden]
        tokens = tokens.reshape(B, (V*L)*tokens.shape[2], tokens.shape[3])  # [B, VL*N, hidden]

        tokens = self.recon(tokens)

        # reshape back to [B*V*L,N,hidden]
        tokens = tokens.view(B, V*L, -1, tokens.shape[-1])
        tokens = tokens.view(B*V*L, -1, tokens.shape[-1])

        feat = self.unpatch(tokens, hw2)   # [B*V*L, hidden, h, w]
        gauss = self.head(feat)            # dict of [B*V*L,*,H,W]
        # reshape to [B,V,L,...]
        out = {}
        for k,vv in gauss.items():
            out[k] = vv.view(B,V,L,*vv.shape[1:])
        return out
