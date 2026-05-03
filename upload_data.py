import os
from huggingface_hub import HfApi

api = HfApi()
token = os.environ.get("HF_TOKEN", "YOUR_HF_TOKEN_HERE")
repo_id = "VarunB2/sketch2face-api"
repo_type = "space"

# Paths to the two specific trained model files locally
cufs_pth = r"c:\Users\ADMIN\OneDrive\Desktop\sketch2face-hybrid-backup\pytorch-CycleGAN-and-pix2pix\checkpoints\cufs_pix2pix\latest_net_G.pth"
celeba_pth = r"c:\Users\ADMIN\OneDrive\Desktop\sketch2face-hybrid-backup\pytorch-CycleGAN-and-pix2pix\checkpoints\cufs_celeba_finetune\latest_net_G.pth"

print("Uploading CUFS model checkpoint...")
if os.path.exists(cufs_pth):
    api.upload_file(
        path_or_fileobj=cufs_pth,
        path_in_repo="checkpoints/cufs_pix2pix/latest_net_G.pth",
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
        commit_message="Upload CUFS trained model checkpoint"
    )
    print("CUFS model uploaded.")
else:
    print("CUFS model not found locally.")

print("Uploading CelebA model checkpoint...")
if os.path.exists(celeba_pth):
    api.upload_file(
        path_or_fileobj=celeba_pth,
        path_in_repo="checkpoints/cufs_celeba_finetune/latest_net_G.pth",
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
        commit_message="Upload CelebA trained model checkpoint"
    )
    print("CelebA model uploaded.")
else:
    print("CelebA model not found locally.")
