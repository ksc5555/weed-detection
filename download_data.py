# Run once: downloads the Plant Seedlings dataset and saves it as data/<species>/*.png
# Dataset: Giselsson et al. (2017), Aarhus University. License CC BY-SA (cite in README).

########################################################################
#                               Imports
########################################################################
# make folders, build file paths
import os
# Hugging Face downloader
from datasets import load_dataset

########################################################################
#                               Settings
########################################################################
# images go in weed-detection/data/
OUTPUT_FOLDER = "data"

########################################################################
#                               Download
########################################################################
# download 5,539 images (~1.7 GB, slow)
dataset = load_dataset("Project-AgML/plant_seedlings_aarhus", split="train")
# species names, e.g. species_names[1] = "charlock"
species_names = dataset.features["label"].names

########################################################################
#                             Save images
########################################################################
# how many images there are in total
total_images = len(dataset)

# image_number = 0, 1, 2, ...   sample = {"image": ..., "label": ...}
for image_number, sample in enumerate(dataset):
    # the picture and its label number (0 to 11)
    picture = sample["image"]
    label_number = sample["label"]
    # turn the label number into a name, e.g. 1 -> "charlock"
    species_name = species_names[label_number]

    # folder for this species, e.g. data/charlock
    species_folder = os.path.join(OUTPUT_FOLDER, species_name)
    # create folder if it doesn't exist
    os.makedirs(species_folder, exist_ok=True)

    # file name padded to 5 digits, e.g. 00012.png
    file_name = f"{image_number:05d}.png"
    image_path = os.path.join(species_folder, file_name)
    picture.save(image_path)

    # show progress every 500 images
    is_progress_step = image_number % 500 == 0
    if is_progress_step:
        print(f"saved {image_number}/{total_images}")

########################################################################
#                               Summary
########################################################################
# should list 12 species folders
created_folders = os.listdir(OUTPUT_FOLDER)
created_folders = sorted(created_folders)
print("done:", created_folders)
