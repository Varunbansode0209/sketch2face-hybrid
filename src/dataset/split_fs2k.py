from pathlib import Path
import random
import shutil


SRC = Path("data/processed/fs2k/photos")
OUT = Path("data/splits/fs2k")
OUT.mkdir(parents=True,exist_ok=True)

gallery_dir = OUT/"gallery"
query_dir = OUT/"query"
gallery_dir.mkdir(parents=True,exist_ok=True)
query_dir.mkdir(parents=True,exist_ok=True)

images = sorted(SRC.glob("*.jpg"))
random.seed(42)
random.shuffle(images)

split = int(0.8 * len(images))
gallery_imgs = images[:split]
query_imgs = images[split:]

for img in gallery_imgs:
    shutil.copy(img,gallery_dir/img.name)

for img in query_imgs:
    shutil.copy(img, query_dir/img.name)


print(f"Gallery:{len(gallery_imgs)}")
print(f"Query:{len(query_imgs)}")