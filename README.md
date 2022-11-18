# One-Class Dense Networks for Anomaly Detection

_Unsupervised learning has been proposed as a tool for model agnostic anomaly detection (AD) in collider physics.  While the goal of these approaches is usually to find events that are `rare' under the Standard Model hypothesis, many approaches are governed by heuristics to strive towards an implicit density estimator.  We study the simplest possible one-class classification method for unsupervised AD and show that it has similar properties to other unsupervised methods.  The method is illustrated using a Gaussian dataset and a simulation of the events at the Large Hadron Collider (LHC).  The simplicity of the one-class classification may enable a deeper understanding of unsupervised AD in the future._

## About this Repository
This repository provides implementations for the one class networks explored in the NeurIPS Machine Learning and the Physical Sciences Workshop paper "One-Class Dense Networks for Anomaly Detection"

Paper: to-be-linked\
Poster: to-be-linked

`notebooks/cpu_code_run_through.ipynb` demos each model created or tested in the paper\
`notebooks/toy_models.ipynb` contains the toy model experiments described in the paper

## Requirements

Cloning this repo:
```
git clone https://github.com/lukasruff/Deep-SVDD-PyTorch.git
```

The easiest way to run this code is through a conda environment
In the root directory of this project:
```
cd <path-to-lhc-oc-classifiers> 
conda create --name <myenv> --file requirements.txt
conda activate <myenv>
```

This will fulfill all requirements needed to run all models except for the Deep-SVDD models

#### `DeepSVDD`

To run the DeepSVDD models, you will need to clone the original Deep-SVDD repository <https://github.com/lukasruff/Deep-SVDD-PyTorch> in the `models/DeepSVDD` folder. You will notice that there are two files already in `models/DeepSVDD`. Once we have cloned the Deep-SVDD repository we need to move this files into the networks folder. (Note that we will be overwriting the main.py file):

Run these commands to set up DeepSVDD
```
cd <path-to-lhc-oc-classifiers> 
cd models/DeepSVDD
git clone https://github.com/lukasruff/Deep-SVDD-PyTorch.git
mv main.py Deep-SVDD-PyTorch/src/networks/main.py
mv fc_net.py Deep-SVDD-PyTorch/src/networks/fc_net.py
cd ../..
```

Additionally, we will have to install more dependencies for DeepSVDD
```
cd models/DeepSVDD/Deep-SVDD-PyTorch
while read requirement; do conda install -n myenv --yes $requirement; done < requirements.txt
cd ../../..
```

Now you should be able to run all source code and notebooks in this repository!
