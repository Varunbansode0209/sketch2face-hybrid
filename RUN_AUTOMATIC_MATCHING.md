# How to Run Automatic CUFS Pairing Fix

## Steps

1. **Activate your conda environment:**
   ```bash
   conda activate sketch2face-hyrbid
   ```

2. **Navigate to the project directory:**
   ```bash
   cd C:\Users\ADMIN\OneDrive\Desktop\sketch2face-hybrid-backup
   ```

3. **Run the automatic matching script:**
   ```bash
   python src/dataset/fix_cufs_pairing_auto.py
   ```

## What the Script Does

1. **Loads all photos and sketches** from `data/raw/cufs/`
2. **Generates embeddings for all photos** using ArcFace
3. **For each sketch:**
   - Converts sketch to photo using Pix2Pix
   - Generates embedding from the generated photo
   - Finds the best matching photo by comparing embeddings
4. **Creates corrected pairing file** at `data/processed/pairs/cufs_fixed.txt`

## Expected Runtime

- Processing 188 photos: ~5-10 minutes
- Processing 188 sketches (with Pix2Pix conversion): ~30-60 minutes
- **Total: Approximately 1-2 hours**

## After Completion

Once the script finishes, you'll see:
- Average similarity scores
- A new file: `data/processed/pairs/cufs_fixed.txt`

Then regenerate the dataset:
```bash
python src/dataset/prepare_pix2pix_cufs.py
python tools/make_pix2pix_pair.py
```

## Troubleshooting

- If you get import errors, make sure the conda environment is activated
- If models are missing, check that `models/onnx/arcface_r50.onnx` and `models/pix2pix/sketch2face.pth` exist
- The script will show progress as it processes each sketch
