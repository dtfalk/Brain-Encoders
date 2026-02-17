# Amygdala Modeling (AMOD)
 
## This project aims to develop image computable models of amygdala activity to better understand the amygdala's role in emotional behavior.

![Schematic of amygdala model development and validation](/AMOD_flowchart.png)

The first part of this project involves building encoding models from existing fMRI data using the ANN EmoNet as a feature extractor. These encoding models aim to predict how the amygdala activates or responds to visual stimuli.
We validate the encoding models by comparing their predictions with observed BOLD responses in independent data from the same participants. They are also validated using naturalistic stimuli from the IAPS and OASIS datasets to see if their predictions align with established associations between amygdala activity and changes in valence and arousal.

The second part of this project involves using the encoding models as targets for activation maximization in a deep generator network to generate artificial stimuli. Artificial stimuli are optimized for a particular region of interest's encoding model. Categorization of artificial stimuli can reveal differences in visual features and in the predicted activations from encoding models.

#### Update: July 24, 2024
This repository contains scripts for the analyses performed in the paper entitled "Understanding human amygdala function with artificial neural networks."

## Dependencies and Related Work
Code for the EmoNet model (Kragel et al., 2019) which was used in encoding model development can be found in [this repository](https://github.com/ecco-laboratory/EmoNet "this repository title").

Instructions for installing CANlab Core Tools which provided many of the functions and Neuroimaging Pattern Masks used in the analyses can be found [here](https://canlab.github.io/_pages/canlab_help_1_installing_tools/canlab_help_1_installing_tools.html "here title").

Instructions for installing SPM12 can be found [here](https://www.fil.ion.ucl.ac.uk/spm/software/spm12/ "here title").

Code for the EmoNet model in PyTorch which was used for artificial stimuli generation can be found in [this repository](https://github.com/ecco-laboratory/emonet-pytorch "this repository title").

Code for the Activation Maximization Optimizer (Wang and Ponce, 2022) used in artificial stimuli generation can be found in [this repository](https://github.com/Animadversio/ActMax-Optimizer-Dev "this repository title").

Instructions for installing MATLAB can be found [here](https://www.mathworks.com/help/install/ug/install-products-with-internet-connection.html "here title").

Instructions for installing PyTorch can be found [here](https://pytorch.org/get-started/locally/ "here title")

## Datasets used for model training and validation
Data relevant to this project can be found in the [OSF repository](https://osf.io/r48gc/ "OSF repository title").

Data from the Naturalistic Neuroimaging Database (Aliko et al., 2020) can be found [here](https://openneuro.org/datasets/ds002837/versions/2.0.0 "here title").

