# from huggingface_hub import snapshot_download

# snapshot_download(
#     repo_id="nvidia/PhysicalAI-SpatialIntelligence-Lyra-SDG",
#     repo_type="dataset",
#     local_dir="lyra_dataset",
#     allow_patterns=[
#         "static/00001/**",
#         "static/00002/**",
#         "static/00003/**",
#         "static/00004/**",
#     ],
#     local_dir_use_symlinks=False
# )


from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="nvidia/PhysicalAI-SpatialIntelligence-Lyra-SDG",
    repo_type="dataset",
    local_dir="lyra_dataset",
    allow_patterns=[
        "static/dataset_part0334.tar",
        "static/dataset_part0335.tar",
        "static/dataset_part0336.tar",
        "static/dataset_part0337.tar",
        "static/dataset_part0338.tar",
    ],
)


cd lyra_dataset/static

for f in dataset_part033{4..8}.tar; do
  echo "Extracting $f"
  tar --no-same-owner -xf "$f"
done


python train_minilyra.py \
  --data-root /workspace/3D_Scene_reconstruction/script/lyra_dataset/static \
  --data-format demo \
  --renderer-backend image \
  --batch-size 1 \
  --views 1 \
  --frames 1 \
  --height 720 \
  --width 1280 \
  --epochs 40 \
  --lr 1e-4 \
  --weight-decay 0 \
  --num-workers 0 \
  --log-every 20 \
  --ckpt-every 500 \
  --save-best
