import torch
from torch.cuda.amp import autocast, GradScaler

def train_one_epoch(model, loader, renderer, optim, device):
    model.train()
    scaler = GradScaler()
    for batch in loader:
        Z = batch["Z"].to(device)                # [B,V,L,C,h,w]
        Ks = batch["K"].to(device)
        Rs = batch["R"].to(device)
        ts = batch["t"].to(device)
        teacher_rgb = batch["rgb"].to(device)    # [B,V,L,3,H,W]
        teacher_depth = batch["depth"].to(device)# [B,V,L,1,H,W]

        optim.zero_grad(set_to_none=True)

        with autocast():
            gauss = model(Z, Ks, Rs, ts)

            # render per (B,V,L)
            B,V,L = Z.shape[:3]
            loss_total = 0.0
            for b in range(B):
                for v in range(V):
                    for t in range(L):
                        # optionally prune
                        # mask = prune_by_opacity({k:gauss[k][b,v,t]...})
                        rgb_pred, depth_pred = renderer.render(
                            {k: gauss[k][b,v,t] for k in gauss},
                            Ks[b,v,t], Rs[b,v,t], ts[b,v,t],
                            teacher_rgb.shape[-2], teacher_rgb.shape[-1]
                        )
                        loss, _ = compute_losses(
                            rgb_pred, teacher_rgb[b,v,t],
                            depth_pred, teacher_depth[b,v,t],
                            gauss["opacity"][b,v,t]
                        )
                        loss_total = loss_total + loss

        scaler.scale(loss_total).backward()
        scaler.step(optim)
        scaler.update()
