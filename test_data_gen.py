import torch
import os
import zipfile
from tqdm import tqdm
import shutil

# Config
base_dir = 'data/test_data'
snap_dir = os.path.join(base_dir, 'snaps')  # Store .pt files here
zip_path = os.path.join(base_dir, 'test_data.zip')

num_samples = 100
img_res_h = 128
img_res_w = 64
img_channels = 2
condition_channels = 10

dry_run = False  # Set to True to skip actual file generation
save_as_zip = True  # Set to True to zip the dataset
cleanup = False  # Set to True to delete snaps/ after zipping

if dry_run:
    print("Dry run mode: No files will be generated.")

# Ensure snaps directory exists
os.makedirs(snap_dir, exist_ok=True)

# Generate and save samples
for i in tqdm(range(num_samples), desc="Generating .pt files"):
    image = torch.randn(img_channels, img_res_h, img_res_w, dtype=torch.float32)
    conditions = torch.randn(condition_channels, img_res_h, img_res_w, dtype=torch.float32)
    if i == 0:
        print(f"Generated sample {i}: image shape {image.shape}, conditions shape {conditions.shape}")
    sample = {'image': image, 'conditions': conditions}
    filename = os.path.join(snap_dir, f'sample_{i:06d}.pt')
    if not dry_run:
        torch.save(sample, filename)

# Optional: zip the snaps folder
if save_as_zip and not dry_run:
    print(f"Packing .pt files into zip: {zip_path}")
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
        for fname in os.listdir(snap_dir):
            full_path = os.path.join(snap_dir, fname)
            arcname = fname  # Store files flat in the zip
            zipf.write(full_path, arcname)
    print(f"Zipped dataset saved at: {zip_path}")

    # Optional cleanup: remove only the snaps folder
    if cleanup:
        shutil.rmtree(snap_dir)
        print(f"Removed snaps folder: {snap_dir}")
