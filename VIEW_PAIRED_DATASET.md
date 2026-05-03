# Where to View the Paired Photo-Sketch Dataset

## 📁 Dataset Locations

### 1. **COMBINED IMAGES (Side-by-Side View) - RECOMMENDED**

These are the easiest to view - each image shows the sketch on the left and photo on the right:

#### Train Set (168 pairs):
- **Location:** `data/pix2pix/train_AB/`
- **Format:** Each file (e.g., `cufs_0001.jpg`) contains sketch|photo side-by-side
- **Size:** 512x256 pixels (256x256 sketch + 256x256 photo)

#### Test Set (20 pairs):
- **Location:** `data/pix2pix/test_AB/`
- **Format:** Each file (e.g., `cufs_0001.jpg`) contains sketch|photo side-by-side
- **Size:** 512x256 pixels (256x256 sketch + 256x256 photo)

**To view:** Simply open any `.jpg` file in these directories with any image viewer!

---

### 2. **SEPARATE FILES (Individual View)**

If you want to view photos and sketches separately, they're organized with matching filenames:

#### Train Set:
- **Photos:** `data/pix2pix/train/photo/` (168 files)
- **Sketches:** `data/pix2pix/train/sketch/` (168 files)
- **Matching:** Files with the same name (e.g., `cufs_0001.jpg`) are paired

#### Test Set:
- **Photos:** `data/pix2pix/test/photo/` (20 files)
- **Sketches:** `data/pix2pix/test/sketch/` (20 files)
- **Matching:** Files with the same name (e.g., `cufs_0001.jpg`) are paired

**To view:** Open corresponding files from both directories to see the pair.

---

## 🔍 Quick View Commands

### Windows Explorer:
```powershell
# Open combined train images
explorer data\pix2pix\train_AB

# Open combined test images
explorer data\pix2pix\test_AB
```

### View Sample Pairs:
```powershell
# View first 5 train pairs
Get-ChildItem data\pix2pix\train_AB\*.jpg | Select-Object -First 5 | ForEach-Object { Start-Process $_.FullName }
```

---

## 📊 Dataset Summary

- **Total Train Pairs:** 168 (within 100-200 range as requested)
- **Total Test Pairs:** 20
- **Source:** CUFS dataset (from `data/raw/cufs/`)
- **Pairing:** Correctly matched using `data/processed/pairs/cufs.txt`
- **Format:** All images are 256x256 pixels (or 512x256 for combined)

---

## 🔄 Regenerate Combined Images

If you need to regenerate the combined images (train_AB and test_AB), run:

```bash
python tools/make_pix2pix_pair.py
```

This will create side-by-side images from the separate photo and sketch directories.
