import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import OpenEXR
import Imath
import zipfile
import io


def _resize_intrinsics(K, src_hw, dst_hw):
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    if src_h == dst_h and src_w == dst_w:
        return K
    sx = dst_w / float(src_w)
    sy = dst_h / float(src_h)
    K = K.copy()
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] *= sx
    K[1, 2] *= sy
    return K

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

class LyraDiffusionOutputDataset(Dataset):

    def __init__(
        self,
        root,
        V=1,
        L=0,
        H=0,
        W=0,
        load_latents=True,
        load_depth=False,
    ):
        self.root = root
        self.V = V
        self.L = L
        self.H = H
        self.W = W
        self.load_latents = load_latents
        self.load_depth = load_depth

        self.sample_dirs = []
        if os.path.isdir(os.path.join(root, "rgb")):
            self.sample_dirs = [root]
        else:
            for d in sorted(os.listdir(root)):
                p = os.path.join(root, d)
                if os.path.isdir(os.path.join(p, "rgb")):
                    self.sample_dirs.append(p)

        if len(self.sample_dirs) == 0:
            raise RuntimeError(f"No diffusion_output folders found under {root}")

        self.samples = []
        for d in self.sample_dirs:
            rgb_dir = os.path.join(d, "rgb")
            for f in sorted(os.listdir(rgb_dir)):
                if f.endswith(".mp4"):
                    sample_id = os.path.splitext(f)[0]
                    self.samples.append((d, sample_id))

        if len(self.samples) == 0:
            raise RuntimeError(f"No mp4 files found under {root}")

    def __len__(self):
        return len(self.samples)

    def _read_video(self, path, max_frames=None):
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

        if len(frames) == 0:
            raise RuntimeError(f"Failed to decode any frames from {path}")

        return np.stack(frames, axis=0)

    def _load_latent(self, path):
        import pickle
        try:
            z = torch.load(path, map_location="cpu")
        except Exception:
            with open(path, "rb") as f:
                z = pickle.load(f)

        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z)

        z = z.float()

        if z.dim() == 5:
            return z
        if z.dim() == 4:
            return z.unsqueeze(0)
        if z.dim() == 3:
            return z.unsqueeze(0).unsqueeze(0)

        raise RuntimeError(f"Unsupported latent shape: {tuple(z.shape)}")

    def __getitem__(self, idx):
        base_dir, sample_id = self.samples[idx]

        rgb_path = os.path.join(base_dir, "rgb", f"{sample_id}.mp4")
        pose_path = os.path.join(base_dir, "pose", f"{sample_id}.npz")
        intr_path = os.path.join(base_dir, "intrinsics", f"{sample_id}.npz")
        latent_path = os.path.join(base_dir, "latent", f"{sample_id}.pkl")

        # ----- Latents -----
        z = None
        if self.load_latents and os.path.exists(latent_path):
            z = self._load_latent(latent_path)
            if self.L == 0 and z.dim() == 5:
                self.L = int(z.shape[1])

        # ----- RGB -----
        rgb_np = self._read_video(rgb_path, max_frames=self.L or None)

        src_h, src_w = rgb_np.shape[1], rgb_np.shape[2]
        tgt_h = self.H if self.H > 0 else src_h
        tgt_w = self.W if self.W > 0 else src_w

        if (tgt_h, tgt_w) != (src_h, src_w):
            rgb_np = np.array([cv2.resize(f, (tgt_w, tgt_h)) for f in rgb_np])

        rgb = torch.from_numpy(rgb_np).float() / 255.0
        rgb = rgb.permute(0, 3, 1, 2)  # [L,3,H,W]
        rgb = rgb.unsqueeze(0)  # [V=1,L,3,H,W]

        # ----- Pose -----
        pose = np.load(pose_path)
        pose_data = pose["data"]
        if self.L > 0:
            pose_data = pose_data[: self.L]

        R = torch.from_numpy(pose_data[:, :3, :3]).float().unsqueeze(0)
        t = torch.from_numpy(pose_data[:, :3, 3]).float().unsqueeze(0)

        # ----- Intrinsics -----
        intr = np.load(intr_path)
        intr_data = intr["data"]
        if self.L > 0:
            intr_data = intr_data[: self.L]

        Ks = []
        for fx, fy, cx, cy in intr_data:
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            K = _resize_intrinsics(K, (src_h, src_w), (tgt_h, tgt_w))
            Ks.append(K)

        K = torch.from_numpy(np.stack(Ks, axis=0)).float().unsqueeze(0)


        # ----- Depth (read EXR from ZIP) -----
        # ----- Depth (read EXR from ZIP safely) -----
        depth_valid = False
        depth = None

        if self.load_depth:
            zip_path = os.path.join(base_dir, "depth", f"{sample_id}.zip")

            if os.path.exists(zip_path):
                depth_list = []

                try:
                    import zipfile
                    import io

                    with zipfile.ZipFile(zip_path, "r") as zip_f:
                        names = set(zip_f.namelist())

                        for frame_idx in range(self.L or rgb.shape[1]):
                            exr_name = f"{frame_idx:05d}.exr"

                            if exr_name not in names:
                                break

                            exr_bytes = zip_f.read(exr_name)

                            # Load EXR from memory
                            exr_file = OpenEXR.InputFile(io.BytesIO(exr_bytes))
                            header = exr_file.header()

                            dw = header["dataWindow"]
                            width = dw.max.x - dw.min.x + 1
                            height = dw.max.y - dw.min.y + 1

                            FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
                            depth_str = exr_file.channel("Z", FLOAT)

                            depth_np = np.frombuffer(depth_str, dtype=np.float32)
                            depth_np = depth_np.reshape((height, width))

                            if (height, width) != (tgt_h, tgt_w):
                                depth_np = cv2.resize(
                                    depth_np,
                                    (tgt_w, tgt_h),
                                    interpolation=cv2.INTER_NEAREST,
                                )

                            depth_list.append(depth_np)

                    if len(depth_list) > 0:
                        depth_np = np.stack(depth_list, axis=0)
                        depth = torch.from_numpy(depth_np).float()

                        # --------------------------------
                        # Normalize depth per-sample
                        # --------------------------------
                        max_val = depth.max()
                        if max_val > 0:
                            depth = depth / (max_val + 1e-6)

                        depth = depth.unsqueeze(1).unsqueeze(0)
                        depth_valid = True

                except Exception as e:
                    print("[WARN] Zip depth read failed:", e)

        if depth is None:
            depth = torch.zeros((1, rgb.shape[1], 1, tgt_h, tgt_w), dtype=torch.float32)



        # ----- Latent Shape Fix -----
        if z is not None:
            if z.dim() == 5 and z.shape[0] != 1:
                z = z[:1]
            if self.L > 0 and z.shape[1] != self.L:
                z = z[:, : self.L]
            z = z.float()

        return {
            "rgb": rgb,
            "depth": depth,
            "depth_valid": depth_valid,
            "K": K,
            "R": R,
            "t": t,
            "z": z,
        }
