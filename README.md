# Team Name - BackPropagators

Kaggle Score - 0.8122

Terrain segmentation inference using MIT-B3 SegFormer model with multi-scale TTA and pseudo labelling training.


## Setup

1. Edit the "TEST_DIR" variable in the paths section with the path to the test images.
(Default value is 'test_images_padded')

## Techniques Used

### 1. Model Architecture
- **SegFormer MIT-B3**: Transformer-based encoder (MiT-B3) with UNet decoder
- **Hybrid Design**: Combines attention mechanisms (global context) + convolutions (local features)

### 2. Training Strategy - Teacher-Student Pseudo-Labeling
- **Teacher Model**: MIT-B2 (smaller model trained first on real data)
- **Pseudo-Label Generation**: Teacher predicts on 1002 test images
- **Student Model**: MIT-B3 (larger model) trained on combined dataset
- **Dataset**: 3174 real images + 1002 pseudo-labeled images = 4176 total

### 3. Loss Function
- **70% Dice Loss**: Optimizes for shape and boundary accuracy
- **30% BCE Loss**: Optimizes for pixel-wise accuracy

### 4. Data Augmentation
- Random horizontal flip (50% probability)
- Color jitter (brightness, contrast, saturation)
- Random crop to 512×512

### 5. Inference Optimizations
- **Multi-Scale TTA**: Inference at scales [0.75, 1.0, 1.25]
- **Horizontal Flip TTA**: Average of original + flipped predictions
- **Mixed Precision**: FP16 inference with `autocast()`

---

## Output

Generates `submission.csv` with RLE-encoded masks:

```csv
image_id,encoded_pixels
test_001,1 5 10 3 25 8 ...
test_002,3 12 50 7 ...
```

## Folder Structure

```
submission/
├── inference_pipeline.py
├── requirements.txt
├── README.md
├── best_student_b3.pth (downloaded)
├── test_images_padded/
│   ├── image1.jpg
│   └── ...
└── submission.csv (generated)
```
