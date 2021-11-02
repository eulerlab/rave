# Removing Inter-Experimental Variability from Functional Data in Systems Neuroscience 

This repository is the official implementation of [https://biorxiv.org/cgi/content/short/2021.10.29.466492v1]. 

![schematic](framework.jpg)

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Downloading the necessary files
In order to reproduce the figures from the paper, you need the following data files:
- Recordings from bipolar cells: path_to_file
- simulated bipolar cell responses: path_to_file
- IPL info files: path_to_file

## Reproducing figures from the paper 

To get started, we suggest running the demo notebook for the simulated data, which loads or trains a model with tuned hyperparameters, runs the evaluation functions and creates the corresponding plots.
The notebook can be found here: ```notebooks/Evaluate_sim_data.ipynb```
You need to adjust the file paths in the corresponding section of the notebook to reflect the locations of the downloaded files.
