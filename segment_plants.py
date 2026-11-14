# Stage 1: find where the plant is in a photo (segmentation), using CMPT 487 methods
#   ExG (greenness) -> median filter (lecture 04) -> LoG blob detection (lecture 05)
# Run each "# %%" cell with "Run Cell" or Shift+Enter. Run download_data.py first.

# %%
########################################################################
#                               Imports
########################################################################
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

# %%
########################################################################
#                          Parameters to tune
########################################################################
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


# %%
########################################################################
#                           Helper functions
########################################################################
def excess_green(image):
    """Greenness image: plants bright, soil dark, scaled 0..1."""
    # keep only the red, green, blue channels (drop transparency if present)
    rgb = image[:, :, :3]
    # use decimals so division is not rounded
    rgb = rgb.astype(float)

    # pull out each colour channel
    red_channel = rgb[:, :, 0]
    green_channel = rgb[:, :, 1]
    blue_channel = rgb[:, :, 2]

    # R+G+B per pixel
    channel_sum = red_channel + green_channel + blue_channel
    # tiny number avoids divide by 0 on black pixels
    channel_sum = channel_sum + 1e-6

    # remove brightness, keep colour
    red = red_channel / channel_sum
    green = green_channel / channel_sum
    blue = blue_channel / channel_sum

    # high when green beats red and blue
    greenness = 2 * green - red - blue

    # rescale so the darkest pixel is 0 and the brightest is 1
    lowest = greenness.min()
    highest = greenness.max()
    value_range = highest - lowest + 1e-6
    rescaled = (greenness - lowest) / value_range
    return rescaled


def find_plants(image):
    """ExG -> median filter -> LoG blobs. Returns (denoised image, blobs)."""
    # colour -> greenness
    greenness = excess_green(image)

    # circular neighbourhood for the median filter
    neighbourhood = disk(MEDIAN_RADIUS)
    # remove speckle, keep edges
    denoised = filters.median(greenness, neighbourhood)

    # find bright patches; each row = (y, x, sigma)
    blobs = feature.blob_log(
        denoised,
        max_sigma=BLOB_MAX_SIGMA,
        num_sigma=10,
        threshold=BLOB_THRESHOLD,
    )
    return denoised, blobs


def show_results(image, denoised, blobs, title=""):
    """Show original, denoised, and blobs side by side."""
    # photo without the transparency channel
    photo = image[:, :, :3]

    # one figure with 3 panels side by side
    figure, panels = plt.subplots(1, 3, figsize=(12, 4))
    original_panel = panels[0]
    greenness_panel = panels[1]
    blobs_panel = panels[2]

    # panel 1: original photo
    original_panel.imshow(photo)
    original_panel.set_title("original")

    # panel 2: greenness after median filter
    greenness_panel.imshow(denoised, cmap="gray")
    greenness_panel.set_title("ExG + median")

    # panel 3: original photo with blob circles
    blob_count = len(blobs)
    blobs_panel.imshow(photo)
    blobs_panel.set_title(f"LoG blobs ({blob_count})")
    for blob in blobs:
        # blob_log gives row (y) first, then column (x), then sigma
        y = blob[0]
        x = blob[1]
        sigma = blob[2]
        # blob radius is about sigma * sqrt(2)
        radius = sigma * np.sqrt(2)
        # Circle wants the centre as (x, y)
        centre = (x, y)
        circle = plt.Circle(centre, radius, color="r", fill=False)
        blobs_panel.add_patch(circle)

    # hide axes
    for panel in panels:
        panel.axis("off")
    figure.suptitle(title)
    plt.show()


# %%
########################################################################
#                           Run on one image
########################################################################
# pattern that matches every image for this species
search_pattern = f"data/{SPECIES}/*.png"
# all matching files, in alphabetical order
image_paths = glob.glob(search_pattern)
image_paths = sorted(image_paths)
# 0 = wrong name or no data yet
image_count = len(image_paths)
print(image_count, "images found for", SPECIES)

# pick one image and load it
image_path = image_paths[IMAGE_INDEX]
image = io.imread(image_path)
# run pipeline
denoised, blobs = find_plants(image)
# display
show_results(image, denoised, blobs, image_path)

# %%
########################################################################
#                       Test on several species
########################################################################
# stage 1 is done when blobs land on plants, not gravel
# two weeds and two crops
species_to_test = ["charlock", "common_wheat", "fat_hen", "maize"]

for species in species_to_test:
    # all images for this species, in alphabetical order
    species_paths = glob.glob(f"data/{species}/*.png")
    species_paths = sorted(species_paths)
    # keep only the first 3
    first_three_paths = species_paths[:3]

    for image_path in first_three_paths:
        image = io.imread(image_path)
        denoised, blobs = find_plants(image)
        show_results(image, denoised, blobs, image_path)
