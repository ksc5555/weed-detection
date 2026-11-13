# Stage 1: find where the plant is in a photo (segmentation), using CMPT 487 methods
#   ExG (greenness) -> median filter (lecture 04) -> LoG blob detection (lecture 05)
# Run each "# %%" cell with "Run Cell" or Shift+Enter. Run download_data.py first.

# %% Imports
# find files by pattern
import glob
# array math
import numpy as np
# plotting
import matplotlib.pyplot as plt
# read images
import skimage.io as io
# median filter
import skimage.filters as filters
# blob_log
import skimage.feature as feature
# circular neighbourhood for median filter
from skimage.morphology import disk

# %% Parameters to tune
# folder inside data/
SPECIES = "charlock"
# which image in that folder
IMAGE_INDEX = 0
# bigger = smoother, but loses thin leaves
MEDIAN_RADIUS = 3
# largest blob size to look for
BLOB_MAX_SIGMA = 30
# lower = more blobs (incl. false ones)
BLOB_THRESHOLD = 0.1


# %% Helper functions
def excess_green(image):
    """Greenness image: plants bright, soil dark, scaled 0..1."""
    # keep RGB only, use decimals
    rgb = image[:, :, :3].astype(float)
    # R+G+B per pixel (tiny number avoids divide by 0)
    channel_sum = rgb.sum(axis=2) + 1e-6
    # remove brightness, keep colour
    red = rgb[:, :, 0] / channel_sum
    green = rgb[:, :, 1] / channel_sum
    blue = rgb[:, :, 2] / channel_sum
    # high when green beats red and blue
    greenness = 2 * green - red - blue
    # rescale to 0..1
    return (greenness - greenness.min()) / (greenness.max() - greenness.min() + 1e-6)


def find_plants(image):
    """ExG -> median filter -> LoG blobs. Returns (denoised image, blobs)."""
    # colour -> greenness
    greenness = excess_green(image)
    # remove speckle, keep edges
    denoised = filters.median(greenness, disk(MEDIAN_RADIUS))
    # each row = (y, x, sigma)
    blobs = feature.blob_log(denoised, max_sigma=BLOB_MAX_SIGMA, num_sigma=10,
                             threshold=BLOB_THRESHOLD)
    return denoised, blobs


def show_results(image, denoised, blobs, title=""):
    """Show original, denoised, and blobs side by side."""
    # 3 panels
    figure, panels = plt.subplots(1, 3, figsize=(12, 4))

    # panel 1: original photo
    panels[0].imshow(image[:, :, :3])
    panels[0].set_title("original")

    # panel 2: greenness after median filter
    panels[1].imshow(denoised, cmap="gray")
    panels[1].set_title("ExG + median")

    # panel 3: original photo with blob circles
    panels[2].imshow(image[:, :, :3])
    panels[2].set_title(f"LoG blobs ({len(blobs)})")
    for y, x, sigma in blobs:
        # blob radius is about sigma * sqrt(2)
        radius = sigma * np.sqrt(2)
        # note: Circle wants (x, y), blob_log gives (y, x)
        panels[2].add_patch(plt.Circle((x, y), radius, color="r", fill=False))

    # hide axes
    for panel in panels:
        panel.axis("off")
    figure.suptitle(title)
    plt.show()


# %% Run on one image
# all images for this species
image_paths = sorted(glob.glob(f"data/{SPECIES}/*.png"))
# 0 = wrong name or no data yet
print(len(image_paths), "images found for", SPECIES)
# load image
image = io.imread(image_paths[IMAGE_INDEX])
# run pipeline
denoised, blobs = find_plants(image)
# display
show_results(image, denoised, blobs, image_paths[IMAGE_INDEX])

# %% Test on several species (Step 1 is done when blobs land on plants, not gravel)
# weeds + crops
for species in ["charlock", "common_wheat", "fat_hen", "maize"]:
    # first 3 images of each species
    for image_path in sorted(glob.glob(f"data/{species}/*.png"))[:3]:
        image = io.imread(image_path)
        denoised, blobs = find_plants(image)
        show_results(image, denoised, blobs, image_path)
