import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

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
    """
    Dataset loader for the demo Lyra diffusion_output format.

    Expected structure:
      root/
        rgb/*.mp4
        pose/*.npz
        intrinsics/*.npz
        latent/*.pkl
        depth/*.zip   (optional, OpenEXR inside)

    You can also pass the parent that contains multiple numbered folders.
    """

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
            # assume root contains multiple subfolders like 0,1,2...
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
        frames = np.stack(frames, axis=0)  # [L,H,W,3]
        return frames

    def _load_depth_zip(self, path, max_frames, target_hw):
        # Avoid OpenEXR warnings if codec isn't enabled in this OpenCV build
        if os.environ.get("OPENCV_IO_ENABLE_OPENEXR", "") not in ("1", "true", "True"):
            return None
        import zipfile
        import tempfile

        Ht, Wt = target_hw
        depths = []
        try:
            with zipfile.ZipFile(path, "r") as z:
                names = sorted(z.namelist())
                if max_frames is not None:
                    names = names[:max_frames]
                for name in names:
                    data = z.read(name)
                    fd, tmp = tempfile.mkstemp(suffix=".exr")
                    os.write(fd, data)
                    os.close(fd)
                    img = cv2.imread(tmp, cv2.IMREAD_UNCHANGED)
                    os.remove(tmp)
                    if img is None:
                        raise RuntimeError("OpenEXR not supported by OpenCV build")
                    if img.ndim == 3:
                        img = img[..., 0]
                    if img.shape[0] != Ht or img.shape[1] != Wt:
                        img = cv2.resize(img, (Wt, Ht), interpolation=cv2.INTER_NEAREST)
                    depths.append(img)
        except Exception:
            return None
        if len(depths) == 0:
            return None
        depth = np.stack(depths, axis=0)  # [L,H,W]
        return depth

    def _load_latent(self, path):
        import warnings
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="You are using `torch.load` with `weights_only=False`",
                    category=FutureWarning,
                )
                z = torch.load(path, map_location="cpu")
        except Exception:
            import pickle
            with open(path, "rb") as f:
                z = pickle.load(f)
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z)
        if isinstance(z, torch.Tensor):
            z = z.float()
        # Normalize shapes to [V,L,C,h,w]
        if z.dim() == 5:
            # If channel-last (V,L,h,w,C), move to (V,L,C,h,w)
            if z.shape[2] > 64 and z.shape[-1] <= 64:
                z = z.permute(0, 1, 4, 2, 3)
            # assume [V,L,C,h,w] or [B,L,C,h,w]
            return z
        if z.dim() == 4:
            # [L,C,h,w]
            return z.unsqueeze(0)
        if z.dim() == 3:
            # [C,h,w] -> [1,1,C,h,w]
            return z.unsqueeze(0).unsqueeze(0)
        raise RuntimeError(f"Unsupported latent shape: {tuple(z.shape)}")

    def __getitem__(self, idx):
        base_dir, sample_id = self.samples[idx]

        rgb_path = os.path.join(base_dir, "rgb", f"{sample_id}.mp4")
        pose_path = os.path.join(base_dir, "pose", f"{sample_id}.npz")
        intr_path = os.path.join(base_dir, "intrinsics", f"{sample_id}.npz")
        latent_path = os.path.join(base_dir, "latent", f"{sample_id}.pkl")
        depth_path = os.path.join(base_dir, "depth", f"{sample_id}.zip")

        # Latents decide L if requested
        z = None
        if self.load_latents and os.path.exists(latent_path):
            z = self._load_latent(latent_path)
            if z.dim() == 5 and self.L == 0:
                self.L = int(z.shape[1])

        # Read RGB
        rgb_np = self._read_video(rgb_path, max_frames=self.L or None)  # [L,H,W,3]
        src_h, src_w = rgb_np.shape[1], rgb_np.shape[2]
        tgt_h = self.H if self.H > 0 else src_h
        tgt_w = self.W if self.W > 0 else src_w
        if (tgt_h, tgt_w) != (src_h, src_w):
            rgb_np = np.array([cv2.resize(f, (tgt_w, tgt_h)) for f in rgb_np])

        rgb = torch.from_numpy(rgb_np).float() / 255.0
        rgb = rgb.permute(0, 3, 1, 2)  # [L,3,H,W]
        rgb = rgb.unsqueeze(0)  # [V=1,L,3,H,W]

        # Poses
        pose = np.load(pose_path)
        pose_data = pose["data"]
        if self.L > 0:
            pose_data = pose_data[: self.L]
        R = torch.from_numpy(pose_data[:, :3, :3]).float()
        t = torch.from_numpy(pose_data[:, :3, 3]).float()
        R = R.unsqueeze(0)  # [V,L,3,3]
        t = t.unsqueeze(0)  # [V,L,3]

        # Intrinsics
        intr = np.load(intr_path)
        intr_data = intr["data"]
        if self.L > 0:
            intr_data = intr_data[: self.L]
        Ks = []
        for fx, fy, cx, cy in intr_data:
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            K = _resize_intrinsics(K, (src_h, src_w), (tgt_h, tgt_w))
            Ks.append(K)
        K = torch.from_numpy(np.stack(Ks, axis=0)).float()
        K = K.unsqueeze(0)

        # Depth (optional)
        depth_valid = False
        depth = None
        if self.load_depth and os.path.exists(depth_path):
            depth_np = self._load_depth_zip(depth_path, self.L or None, (tgt_h, tgt_w))
            if depth_np is not None:
                depth = torch.from_numpy(depth_np).float()
                depth = depth.unsqueeze(1)  # [L,1,H,W]
                depth = depth.unsqueeze(0)  # [V,L,1,H,W]
                depth_valid = True
        if depth is None:
            depth = torch.zeros((1, rgb.shape[1], 1, tgt_h, tgt_w), dtype=torch.float32)

        # Latents
        if z is not None:
            if z.dim() == 5:
                # ensure V=1
                if z.shape[0] != 1:
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
