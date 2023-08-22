# TL_FS_C
#Experiment Activation

This document provides instructions on how to activate and run the machine learning experiment using the provided code. 
The experiment involves training and evaluating Siamese Audio Spectrogram Transformer models. 
Follow the steps below to conduct the experiments and analyze the results.

## Prerequisites

- Python 3.x environment with required dependencies installed (specified in the code).
- Access to the necessary dataset files (downloadable from the provided GitHub repository).

## Code Overview

The code is organized into distinct sections that can be activated using conditional flags. Each section corresponds to a specific experiment or functionality.

### Experiment Sections

1. **SIEAMISE**: Train and test SIEAMIS models.
    - Set this flag to `True` to perform the SIEAMIS training and testing experiments.
    - Train Siamese AST models with varying hyperparameters and configurations.
    
2. **CHACK_SIEAMIS_TRAIN**: Analyze and visualize SIEAMIS model performance.
    - Set this flag to `True` to analyze and visualize the trained SIEAMIS models' performance.
    - Calculate and plot metrics like balanced accuracy and multi-class accuracy for different models.

## Data Preparation

1. If you don't have the required datasets locally, it will download them from the provided GitHub repository.
2. Update the data paths in the code to match your local file structure.

## Running the Experiments

1. Activate the desired experiment sections by setting their respective flags to `True` or `False`.
2. Ensure that the necessary dataset files are available locally.
3. Execute the script. The activated sections will perform the specified experiments.
4. The script will generate metrics and visualizations based on the activated sections.

## Customization
1. Adjust hyperparameters and configurations in the parameter dictionaries (`parser_param` and `parser_param1`).
2. Modify paths to dataset files, experiment directories, and model checkpoints as per your file structure.
3. Extend or modify the script to accommodate additional requirements or variations.
