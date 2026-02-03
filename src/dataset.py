import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

def read_video_mp4(path, max_frames=None):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    frames = np.stack(frames, axis=0)  # [L,H,W,3]
    return frames

class LyraTeacherSubset(Dataset):
    """
    Make this match your extracted subset layout.
    Keep it dumb + explicit; avoid clever magic early.
    """
    def __init__(self, root, split="train", V=1, L=1, H=176, W=320, load_latents=True):
        self.root = root
        self.split = split
        self.V = V
        self.L = L
        self.H, self.W = H, W
        self.load_latents = load_latents

        # Example: a list of sample folders
        # root/train/sample_000123/{rgb.mp4, depth.npz, pose.npz, latents.npz?}
        split_dir = os.path.join(root, split)
        self.samples = sorted([os.path.join(split_dir, d) for d in os.listdir(split_dir)])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir = self.samples[idx]

        # ---- RGB frames ----
        # You might have per-trajectory mp4s; here assume rgb_v0.mp4, rgb_v1.mp4...
        rgbs = []
        for v in range(self.V):
            vid_path = os.path.join(sample_dir, f"rgb_v{v}.mp4")
            frames = read_video_mp4(vid_path, max_frames=self.L)  # [L,H,W,3]
            frames = np.array([cv2.resize(f, (self.W, self.H)) for f in frames])
            frames = torch.from_numpy(frames).float() / 255.0       # [L,H,W,3]
            frames = frames.permute(0, 3, 1, 2)                     # [L,3,H,W]
            rgbs.append(frames)
        rgb = torch.stack(rgbs, dim=0)  # [V,L,3,H,W]

        # ---- Depth ----
        # Example: depth stored as npz with key "depth" shaped [V,L,H,W] or similar
        depth_path = os.path.join(sample_dir, "depth.npz")
        depth_np = np.load(depth_path)["depth"]
        # adapt shape as needed:
        depth_np = depth_np[:self.V, :self.L]
        depth = torch.from_numpy(depth_np).float()                # [V,L,H,W]
        depth = depth.unsqueeze(2)                                # [V,L,1,H,W]

        # ---- Poses ----
        pose_path = os.path.join(sample_dir, "pose.npz")
        pose = np.load(pose_path)
        K = torch.from_numpy(pose["K"][:self.V, :self.L]).float()  # [V,L,3,3]
        R = torch.from_numpy(pose["R"][:self.V, :self.L]).float()  # [V,L,3,3]
        t = torch.from_numpy(pose["t"][:self.V, :self.L]).float()  # [V,L,3]

        # ---- Latents (optional) ----
        z = None
        if self.load_latents:
            lat_path = os.path.join(sample_dir, "latents.npz")
            if os.path.exists(lat_path):
                z_np = np.load(lat_path)["z"]   # expected [V,L,C,h,w]
                z = torch.from_numpy(z_np[:self.V, :self.L]).float()

        return {"rgb": rgb, "depth": depth, "K": K, "R": R, "t": t, "z": z}
