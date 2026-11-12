# Run once: downloads the Plant Seedlings dataset and saves it as data/<species>/*.png
# Dataset: Giselsson et al. (2017), Aarhus University. License CC BY-SA (cite in README).

# make folders, build file paths
import os
# Hugging Face downloader
from datasets import load_dataset

# images go in weed-detection/data/
OUT_DIR = "data"

# download 5,539 images (~1.7 GB, slow)
ds = load_dataset("Project-AgML/plant_seedlings_aarhus", split="train")
# species names, e.g. names[1] = "charlock"
names = ds.features["label"].names

# i = image number, row = {image, label}
for i, row in enumerate(ds):
    # e.g. data/charlock
    folder = os.path.join(OUT_DIR, names[row["label"]])
    # create folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    # save as 00012.png
    row["image"].save(os.path.join(folder, f"{i:05d}.png"))
    # show progress every 500 images
    if i % 500 == 0:
        print(f"saved {i}/{len(ds)}")

# should list 12 species folders
print("done:", sorted(os.listdir(OUT_DIR)))
