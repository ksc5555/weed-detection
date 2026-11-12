# Step 1: find plants in a photo using CMPT 487 methods
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
import skimage.filters as filt
# blob_log
import skimage.feature as feat
# circular neighbourhood for median filter
from skimage.morphology import disk

# %% Parameters to tune
# folder inside data/
SPECIES = "charlock"
# which image in that folder
IMG_INDEX = 0
# bigger = smoother, but loses thin leaves
MEDIAN_RADIUS = 3
# largest blob size to look for
LOG_MAX_SIGMA = 30
# lower = more blobs (incl. false ones)
LOG_THRESHOLD = 0.1


# %% Helper functions
def excess_green(img):
    """Greenness image: plants bright, soil dark, scaled 0..1."""
    # keep RGB only, use decimals
    img = img[:, :, :3].astype(float)
    # R+G+B per pixel (avoid divide by 0)
    s = img.sum(axis=2) + 1e-6
    # remove brightness, keep colour
    r, g, b = img[:, :, 0] / s, img[:, :, 1] / s, img[:, :, 2] / s
    # high when green beats red and blue
    exg = 2 * g - r - b
    # rescale to 0..1
    return (exg - exg.min()) / (exg.max() - exg.min() + 1e-6)


def find_plants(img):
    """ExG -> median filter -> LoG blobs. Returns (filtered image, blobs)."""
    # colour -> greenness
    exg = excess_green(img)
    # remove speckle, keep edges
    exg_d = filt.median(exg, disk(MEDIAN_RADIUS))
    # each row = (y, x, sigma)
    blobs = feat.blob_log(exg_d, max_sigma=LOG_MAX_SIGMA, num_sigma=10,
                          threshold=LOG_THRESHOLD)
    return exg_d, blobs


def show(img, exg_d, blobs, title=""):
    """Show original, filtered, and blobs side by side."""
    # 3 panels
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(img[:, :, :3]); ax[0].set_title("original")
    ax[1].imshow(exg_d, cmap="gray"); ax[1].set_title("ExG + median")
    ax[2].imshow(img[:, :, :3]); ax[2].set_title(f"LoG blobs ({len(blobs)})")
    # one red circle per blob
    for y, x, sig in blobs:
        # note: Circle wants (x, y), blob_log gives (y, x)
        ax[2].add_patch(plt.Circle((x, y), sig * np.sqrt(2), color="r", fill=False))
    # hide axes
    for a in ax:
        a.axis("off")
    fig.suptitle(title)
    plt.show()


# %% Run on one image
# all images for this species
paths = sorted(glob.glob(f"data/{SPECIES}/*.png"))
# 0 = wrong name or no data yet
print(len(paths), "images found for", SPECIES)
# load image
img = io.imread(paths[IMG_INDEX])
# run pipeline
exg_d, blobs = find_plants(img)
# display
show(img, exg_d, blobs, paths[IMG_INDEX])

# %% Test on several species (Step 1 is done when blobs land on plants, not gravel)
# weeds + crops
for sp in ["charlock", "common_wheat", "fat_hen", "maize"]:
    # first 3 of each
    for p in sorted(glob.glob(f"data/{sp}/*.png"))[:3]:
        im = io.imread(p)
        e, bl = find_plants(im)
        show(im, e, bl, p)
