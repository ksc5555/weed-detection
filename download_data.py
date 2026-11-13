# Run once: downloads the Plant Seedlings dataset and saves it as data/<species>/*.png
# Dataset: Giselsson et al. (2017), Aarhus University. License CC BY-SA (cite in README).

# make folders, build file paths
import os
# Hugging Face downloader
from datasets import load_dataset

# images go in weed-detection/data/
OUTPUT_FOLDER = "data"

# download 5,539 images (~1.7 GB, slow)
dataset = load_dataset("Project-AgML/plant_seedlings_aarhus", split="train")
# species names, e.g. species_names[1] = "charlock"
species_names = dataset.features["label"].names

# image_number = 0, 1, 2, ...   sample = {"image": ..., "label": ...}
for image_number, sample in enumerate(dataset):
    # e.g. data/charlock
    species_folder = os.path.join(OUTPUT_FOLDER, species_names[sample["label"]])
    # create folder if it doesn't exist
    os.makedirs(species_folder, exist_ok=True)
    # save as 00012.png
    image_path = os.path.join(species_folder, f"{image_number:05d}.png")
    sample["image"].save(image_path)
    # show progress every 500 images
    if image_number % 500 == 0:
        print(f"saved {image_number}/{len(dataset)}")

# should list 12 species folders
print("done:", sorted(os.listdir(OUTPUT_FOLDER)))
