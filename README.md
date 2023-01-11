# Removing Inter-Experimental Variability from Functional Data in Systems Neuroscience 

This repository is the official implementation of [https://proceedings.neurips.cc/paper/2021/hash/1e5eeb40a3fce716b244599862fd2200-Abstract.html]. 

![schematic](framework.jpg)

## Getting started

You can use the code in this repository to reproduce the figures from our paper by following the instructions below.
1. ```git clone``` the repository onto your machine
2. Have a look at the packages listed in the requirements file; make sure installing the packages won't mess with your python installation. Then: 
3. From within the directory containing the ```requirements.txt``` file, run ```pip install -r requirements.txt```

## Downloading the necessary files
You can download the data we worked with from here: https://doi.org/10.12751/g-node.5iije0 
In order to reproduce the figures from the paper, you need the following data files:
- Recordings from bipolar cells: bio/...
- simulated bipolar cell responses: silico/...
- IPL info files: ipl/...
- 
## Reproducing figures from the paper 

To get started, we suggest running the demo notebook for the simulated data, which loads or trains a model with tuned hyperparameters, runs the evaluation functions and creates the corresponding plots.
The notebook can be found here: ```notebooks/Evaluate_sim_data_template.ipynb```
You need to adjust the file paths in the corresponding section of the notebook to reflect the locations of the downloaded files.
