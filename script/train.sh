# to check if dataset is correct:

# python train.py \
#   --data-root /workspace/3D_Scene_reconstruction/script/lyra_dataset/static \
#   --data-format teacher \
#   --renderer-backend image \
#   --batch-size 1 \
#   --views 1 \
#   --frames 1 \
#   --epochs 50 \
#   --lr 3e-4 \
#   --weight-decay 0 \
#   --log-every 50 \
#   --ckpt-every 500 \
#   --save-best


python train_minilyra.py   --data-root /workspace/3D_Scene_reconstruction/script/lyra_dataset/static   --data-format demo   --renderer-backend image   --batch-size 1   --views 1   --frames 3   --epochs 80   --lr 2e-4   --weight-decay 0   --num-workers 0   --log-every 50   --ckpt-every 500   --save-best


python train_minilyra.py \
  --data-root ... \
  --data-format demo \
  --head-mode points \
  --renderer-backend gsplat \
  --num-points 20000 \
  --frames 3 --height 704 --width 1280 \
  --epochs 20 --lr 1e-4 --hidden 384 \
  --ckpt-every 0 --save-best