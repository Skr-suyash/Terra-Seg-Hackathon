"""
TerraSeg: Inference Pipeline with Multi-Scale TTA
Downloads MIT-B3 model from Google Drive and generates submission.csv
Matches notebook_best.ipynb exactly for RLE encoding
"""

import os
import glob
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
import torch.nn.functional as F
import gdown

# Install dependencies if needed
try:
    import segmentation_models_pytorch as smp
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch"])
    import segmentation_models_pytorch as smp

# =============================================================================
# CONFIGURATION
# =============================================================================

# Google Drive link for model weights
GDRIVE_FILE_LINK = 'https://drive.google.com/file/d/1fOWBVYQk4-Nh1DGif2mDVRGLg8SlbVyu/view?usp=sharing'

# Paths
WEIGHTS_PATH = 'best_student_b3.pth'
TEST_DIR = 'test_images_padded'
OUTPUT_FILE = 'submission.csv'

# TTA Configuration (matches notebook)
TTA_SCALES = [0.75, 1.0, 1.25]

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# =============================================================================
# DOWNLOAD MODEL
# =============================================================================

if not os.path.exists(WEIGHTS_PATH):
    print(f"📥 Downloading weights from Google Drive...")
    print(f"   From: {GDRIVE_FILE_LINK}")
    print(f"   To: {WEIGHTS_PATH}")
    gdown.download(GDRIVE_FILE_LINK, WEIGHTS_PATH, quiet=False, fuzzy=True, use_cookies=False)
    
    if os.path.exists(WEIGHTS_PATH):
        file_size = os.path.getsize(WEIGHTS_PATH) / (1024 * 1024)
        if file_size < 10:
            print(f"⚠️ Downloaded file is only {file_size:.1f} MB - likely corrupted!")
            os.remove(WEIGHTS_PATH)
        else:
            print(f"✓ Downloaded to: {WEIGHTS_PATH} ({file_size:.1f} MB)")
else:
    print(f"✓ Weights already exist at: {WEIGHTS_PATH}")

print("=" * 60)
print("TerraSeg: Inference Pipeline")
print("=" * 60)

# =============================================================================
# MODEL (matches notebook exactly)
# =============================================================================

def get_student_model():
    """Returns the exact model structure used in Student Training."""
    model = smp.Unet(
        encoder_name="mit_b3",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        decoder_use_batchnorm=True,
    )
    return model

# =============================================================================
# RLE ENCODING (CRITICAL: must use order="F" for Kaggle!)
# =============================================================================

def rle_encode(mask):
    """Encodes a binary mask to RLE format for Kaggle.
    IMPORTANT: Uses Fortran order (column-major) as required by Kaggle!
    """
    pixels = mask.flatten(order="F")  # <-- CRITICAL: Fortran order!
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# =============================================================================
# MULTI-SCALE TTA (matches notebook exactly)
# =============================================================================

def predict_multiscale_tta(model, image_tensor, scales=TTA_SCALES):
    """Runs inference at multiple scales with horizontal flip TTA."""
    b, c, h, w = image_tensor.shape
    final_output = torch.zeros((b, 1, h, w), device=device)
    
    for scale in scales:
        if scale != 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            new_h = int(np.ceil(new_h / 32) * 32)
            new_w = int(np.ceil(new_w / 32) * 32)
            
            input_scaled = F.interpolate(
                image_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False
            )
        else:
            input_scaled = image_tensor

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pred = torch.sigmoid(model(input_scaled))
                pred_flip = torch.sigmoid(model(torch.flip(input_scaled, dims=[3])))
                pred_flip = torch.flip(pred_flip, dims=[3])
        
        pred_avg = (pred + pred_flip) / 2.0
        
        if scale != 1.0:
            pred_avg = F.interpolate(
                pred_avg, size=(h, w), mode='bilinear', align_corners=False
            )
            
        final_output += pred_avg

    final_output /= len(scales)
    return final_output

# =============================================================================
# MAIN INFERENCE
# =============================================================================

def run_inference():
    if not os.path.exists(WEIGHTS_PATH):
        print(f"❌ Critical Error: '{WEIGHTS_PATH}' not found!")
        return

    # Load Model
    print("⚙️ Loading Student Model (mit_b3)...")
    model = get_student_model()
    
    # Load Weights
    print(f"📖 Reading weights from {WEIGHTS_PATH}...")
    state_dict = torch.load(WEIGHTS_PATH, map_location=device, weights_only=False)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    print("✅ Student Weights Loaded Successfully.")

    # Get test files
    test_dir = TEST_DIR
    if not os.path.exists(test_dir):
        test_dir = "test_images"
        print(f"⚠️ Using fallback: {test_dir}")
    
    test_files = sorted(glob.glob(os.path.join(test_dir, "*.*")))
    print(f"🚀 Processing {len(test_files)} images using Multi-Scale TTA {TTA_SCALES}...")
    
    results = []
    
    for img_path in tqdm(test_files):
        try:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # Load image
            image = Image.open(img_path).convert("RGB")
            
            # Preprocess
            input_tensor = T.functional.to_tensor(image).unsqueeze(0).to(device)
            input_tensor = T.functional.normalize(input_tensor, 
                                                  mean=[0.485, 0.456, 0.406], 
                                                  std=[0.229, 0.224, 0.225])
            
            # Multi-Scale TTA
            pred_mask = predict_multiscale_tta(model, input_tensor, scales=TTA_SCALES)
            
            # Binarize
            pred_mask_np = (pred_mask > 0.5).float().cpu().numpy().astype(np.uint8)[0, 0]
            
            # RLE Encode
            rle = rle_encode(pred_mask_np)
            results.append({'image_id': img_name, 'encoded_pixels': rle})
            
        except Exception as e:
            print(f"⚠️ Error on {img_name}: {e}")
            
    # Save Submission
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n✅ INFERENCE COMPLETE!")
    print(f"📊 Processed {len(results)} images.")
    print(f"💾 Saved to: {OUTPUT_FILE}")
    
    # Show file size
    file_size = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"📦 File size: {file_size:.1f} MB")

if __name__ == "__main__":
    run_inference()
