from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="nvidia/PhysicalAI-SpatialIntelligence-Lyra-SDG",
    repo_type="dataset",
    local_dir="lyra_dataset",
    allow_patterns=[
        "static/00001/**",
        "static/00002/**",
        "static/00003/**",
        "static/00004/**",
    ],
    local_dir_use_symlinks=False
)
