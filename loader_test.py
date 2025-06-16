import torch
from torch.utils.data import DataLoader
from training.dataset_cond import PTDataset  # Update with actual filename

# === Configuration ===
DATA_PATH = 'test_data/test_data.zip'  # or 'test_data/snaps'
IMG_RES_H = 128
IMG_RES_W = 64
IMG_CHANNELS = 2
COND_CHANNELS = 10
BATCH_SIZE = 8
USE_GPU = True  # Set to False if testing on CPU
XFLIP = True
YFLIP = True
CACHE = True
MAX_MISMATCHES = 0

# === Instantiate dataset ===
dataset = PTDataset(
    path=DATA_PATH,
    img_res_h=IMG_RES_H,
    img_res_w=IMG_RES_W,
    img_channels=IMG_CHANNELS,
    cond_channels=COND_CHANNELS,
    xflip=XFLIP,
    yflip=YFLIP,
    cache=CACHE,
    max_mismatches=MAX_MISMATCHES
)

print(f"✅ Dataset loaded: {len(dataset)} samples (including flips)")

# === Create DataLoader ===
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

# === Test a few batches ===
device = torch.device('cuda' if USE_GPU and torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")
print(torch.cuda.is_available())

for batch_idx, (images, conditions) in enumerate(loader):
    print(f"\n🔹 Batch {batch_idx}")
    print(f"  Image shape     : {images.shape}")      # [B, C, H, W]
    print(f"  Condition shape : {conditions.shape}")  # [B, D, H, W]

    images = images.to(device, non_blocking=True)
    conditions = conditions.to(device, non_blocking=True)

    print(f"  → Moved to {device}")
    
    if batch_idx >= 2:  # test only a few batches
        break

dataset.close()
print("✅ Dataset test completed.")
