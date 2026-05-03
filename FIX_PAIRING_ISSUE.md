# Fixing CUFS Pairing Issue

## Problem Identified

The pairing file `data/processed/pairs/cufs.txt` was created by simply matching photos and sketches by alphabetical index, which **does not guarantee they are the same person**. This is why you're seeing mismatched pairs in `cufs_0001.jpg` and other images.

## Solution Options

### Option 1: Automatic Matching (Recommended but Slow)

Use face recognition to automatically match sketches to photos:

```bash
python src/dataset/fix_cufs_pairing_auto.py
```

**How it works:**
1. Converts each sketch to a photo using Pix2Pix
2. Generates face embeddings for all photos and generated photos
3. Matches each sketch to the photo with highest similarity
4. Creates `data/processed/pairs/cufs_fixed.txt`

**Note:** This will take a while (188 sketches × processing time), but will create correct pairs.

### Option 2: Manual Verification

If automatic matching doesn't work well, you can manually verify and fix pairs:

1. Open `data/processed/pairs/cufs.txt`
2. For each line, verify that the photo and sketch show the same person
3. Fix incorrect pairs manually
4. Save as `data/processed/pairs/cufs_fixed.txt`

### Option 3: Use CUFS Dataset Documentation

Check if the CUFS dataset has official pairing information or documentation that specifies how photos and sketches should be matched.

## After Fixing

Once you have `data/processed/pairs/cufs_fixed.txt`:

1. The `prepare_pix2pix_cufs.py` script will automatically use it
2. Regenerate the dataset:
   ```bash
   python src/dataset/prepare_pix2pix_cufs.py
   python tools/make_pix2pix_pair.py
   ```

## Current Status

- ❌ Original pairing file has incorrect pairs (matched by index, not identity)
- ✅ Script created to automatically fix pairs using face recognition
- ✅ Prepare script updated to use fixed pairing file if available
- ⏳ Need to run automatic matching or manually fix pairs

## Quick Test

To verify if a pair is correct, visually check:
- `data/pix2pix/train_AB/cufs_0001.jpg` - Does the sketch and photo show the same person?

If not, the pairing needs to be fixed using one of the options above.
