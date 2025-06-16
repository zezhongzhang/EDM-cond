import torch
import os
from tqdm import tqdm

# Config
output_dir = 'test_data'
num_samples = 100  # Change this to generate more or fewer samples
img_res_h = 128  # Height of the image
img_res_w = 64   # Width of the image
img_channels = 2  # Number of channels in the image
condition_channels = 10  # Number of condition channels

dry_run = False  # Set to True to skip actual file generation
if dry_run:
    print("Dry run mode: No files will be generated.")

# Make sure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Generate and save samples
for i in tqdm(range(num_samples), desc="Generating .pt files"):
    image = torch.randn(img_channels, img_res_h, img_res_w , dtype=torch.float32)
    conditions = torch.randn(condition_channels, img_res_h, img_res_w , dtype=torch.float32)
    if i == 0:
        print(f"Generated sample {i}: image shape {image.shape}, conditions shape {conditions.shape}")
    sample = {
        'image': image,
        'conditions': conditions
    }
    filename = os.path.join(output_dir, f'sample_{i:06d}.pt')
    if not dry_run:
        torch.save(sample, filename)  
