from huggingface_hub import HfApi

api = HfApi()
files = api.list_repo_files("nvidia/PhysicalAI-SpatialIntelligence-Lyra-SDG", repo_type="dataset")

print("num files:", len(files))
# show likely shard files (tar files)
tar_files = [f for f in files if f.endswith(".tar")]
print("num tar files:", len(tar_files))
print("\n".join(tar_files[:50]))  # peek first 50
