# Weed Detection

Detecting crop and weed seedlings in field images using classical image processing
(Excess Green index, median filtering, Laplacian of Gaussian blob detection).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python download_data.py
```

Then open `explore_step1.py` in VS Code and run the `# %%` cells.

## Status

- [ ] Step 1: locate plants with ExG + median filter + LoG blobs
- [ ] Step 2: colour edge features to outline leaves
- [ ] Step 3: classify crop vs. weed

## Dataset

Plant Seedlings Dataset (Aarhus University), via Hugging Face
(`Project-AgML/plant_seedlings_aarhus`). Licensed CC BY-SA.

> Giselsson, T. M., Jørgensen, R. N., Jensen, P. K., Dyrmann, M., & Midtiby, H. S. (2017).
> A Public Image Database for Benchmark of Plant Seedling Classification Algorithms.
> arXiv:1711.05458.
