# eeg-topomap-right-or-left

EEG topographic map visualization for left and right hand motor imagery tasks using MNE-Python.

## Overview

This repository provides scripts to visualize cortical activation patterns during left and right hand motor tasks using EEG topographic maps. It uses the [MNE-Python](https://mne.tools/) library to load, process, and visualize EEG data from the PhysioNet EEG Motor Movement/Imagery Dataset.

## Features

- **Motor Imagery Classification** (`left_or_right.py`): Extracts PSD features from EEG data and classifies left vs. right hand motor imagery using scikit-learn.
- **Topographic Map Visualization** (`topomap.py`): Generates topographic maps showing spatial distribution of alpha-band power during motor tasks.

## Requirements

- Python 3.8+
- MNE-Python
- NumPy
- scikit-learn
- Matplotlib

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Classify left vs. right hand motor imagery
python left_or_right.py

# Generate topographic maps
python topomap.py
```

The scripts automatically download sample EEG data from the PhysioNet dataset via `mne.datasets.eegbci`.

## Output

- `topomap_result.png`: Topographic map visualization of motor imagery patterns.
