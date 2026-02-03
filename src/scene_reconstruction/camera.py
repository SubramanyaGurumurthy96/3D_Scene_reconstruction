import torch
import torch.nn as nn

class PluckerEncoder(nn.Module):
    def __init__(self, in_ch=6, out_ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(64, out_ch, 3, stride=2, padding=1),
        )

    def forward(self, E_hw6):
        """
        E_hw6: [B,6,H,W]
        returns: [B,out_ch,h,w]
        """
        return self.net(E_hw6)
