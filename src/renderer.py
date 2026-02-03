class GaussianRenderer:
    def __init__(self, backend):
        self.backend = backend  # gsplat / diff-gaussian-rasterization

    def render(self, gauss, K, R, t, H, W):
        """
        gauss: dict pos/scale/quat/opacity/rgb shaped [N, ...] or [H,W,...] depending on backend
        Return: rgb [3,H,W], depth [1,H,W]
        """
        # TODO: implement with chosen backend
        raise NotImplementedError
