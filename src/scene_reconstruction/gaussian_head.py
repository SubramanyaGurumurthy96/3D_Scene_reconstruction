
import torch
import torch.nn as nn
import torch.nn.functional as F

def normalize_quat(q):
    return q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)

class GaussianHead(nn.Module):
    def __init__(self, hidden, out_ch=14):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(hidden, hidden, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(hidden, hidden//2, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(hidden//2, out_ch, 1),
        )

    def forward(self, feat):
        """
        feat: [B, hidden, h2, w2]
        returns dict with shapes [B,H,W,*]
        """
        x = self.deconv(feat)          # [B,14,H,W]
        pos   = x[:, 0:3]
        scale = F.softplus(x[:, 3:6]) + 1e-4
        quat  = normalize_quat(x[:, 6:10].permute(0,2,3,1)).permute(0,3,1,2)
        opacity = torch.sigmoid(x[:, 10:11])
        rgb   = torch.sigmoid(x[:, 11:14])
        return {"pos": pos, "scale": scale, "quat": quat, "opacity": opacity, "rgb": rgb}
