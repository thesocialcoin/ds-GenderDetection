# ds-GenderDetection

Creation of the gender classification model [MiniAM2](https://huggingface.co/CitibeatsAI/miniam2). Inputs are _screen_name_, _name_ and _bio_ description of Twitter users and the outputs are "man", "woman" and "organization" classes. 

## Languages available: 
EN, ES, FR

## Goal of this project 

Can be structured in two steps:
1. **Soft labeled training set creation:** Soft label an unlabeled set given a set of labeling functions and potentially an annotated data of around 500 entries per language.
2. **Build,Train and evaluate MiniAM2**: Easily build the model and call the fit method sending the soft labeled training set of previous step. Furthermore, easily evaluate your trained model in your test set

### Further goals

Provide the methodology to replicate MiniAM2 and make available the option of contributing and improveming on the model

## Get started

### Installation

Coming soon...

### Test the environment and model building

Open notebook with title _1.Test model loading, adapting tokenizers and prediction methods.ipynb_ and follow the walkthrough instructions

### Train MiniAM2

We offer an example of MiniAM2 training in the notebook titled _2.Train MiniAM2.ipynb_. A training set must be provided first with the path of the location.

### Evaluate MiniAM2

Evaluate your model using as an example the notebook _3.Compute Model Metrics.ipynb_. The notebook assumes that some test sets exist in the project and whose path is given in the notebook.
