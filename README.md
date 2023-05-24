# ds-GenderDetection

Creation of the gender classification model [MiniAM2](https://huggingface.co/CitibeatsAI/miniam2).

**License:** This software is © 2023 The Social Coin, SL and is licensed under the OPEN RAIL M License. See [license](https://www.citibeats.com/open-rail-m-license-for-citibeats)

The latest version of the model is uploaded and open to use in 
[Hugging Face](https://huggingface.co/CitibeatsAI/miniam2) hub. We strongly recommend to visit the model card in the
link, to meet, before the usage of the model, the limitations, bias and good and bad uses of the MiniAM2.

## Inputs/Outputs

Inputs are _screen_name_, _name_ and _bio_ description of Twitter users and the outputs are _organization_, _man_ and 
_woman_  classes. Respectively

## Languages available: 
EN, ES, FR

## Goal of this project 

Can be structured in two steps:
1. **Soft labeled training set creation:** Soft label an unlabeled set given a set of labeling functions and potentially an annotated data of around 500 entries per language.
2. **Build,Train and evaluate MiniAM2**: Easily build the model and call the fit method sending the soft labeled training set of previous step. Furthermore, easily evaluate your trained model in your test set

### Further goals

Provide the methodology to replicate MiniAM2 and make available the option of contributing and improving on the model:
- Adding languages
- Improving metrics
- Provide labeling functions
- Modify and improve methodology

## Data 

Right now, data is not shared in the project. However, we provide the steps to obtain a labeled set if an unlabeled
training set is given. There is no need to provide an annotated dataset, since we share the decision functions 
models for english, spanish and french

## Get started

### Installation

Create a new virtual environment and activate it

```bash
python -m venv miniam2
source miniam2/bin/activate
```

Then install all dependencies specified in _requirements.txt_
```bash
pip install -r requirements.txt
```

That is all! You can now try to run the notebooks to check all the functionality of the project.

### Create a training set using Assemblage distillation
Notebooks 1,2,3,4 and 5 show the complete steps needed to obtain a soft labeled training set departing from an unlabeled
training set by means of the Assemblage distillation process. 

*Remarks*
- Notebooks 1 and 2 can be skipped
- Notebook 3 can also be skipped and used decision functions for english, spanish and french languages given in 
_models_ folder named _LogisticRegression[lang]_, where _lang_ stands for the language code. We claim in our paper that 
decision functions given can be applied to some other languages without losing quality.
- Notebook 3 needs an annotated dataset (called dev set) externally provided to successfully train a decision function. 
- Notebook 4 needs an unlabeled training set. The resulting training set must be saved.

### Train MiniAM2 and evaluate it with several metrics
Notebooks 5,6,7 and 8 walk the steps needed to train a MiniAM2 model assuming a training set is available. We offer an 
example of MiniAM2 training in the notebook 6. A training set must be provided first with the path of the location.
Evaluate your model using as an example the notebook 7 and 8. 
The notebook assumes that some test sets exist in the project and whose path is given in the notebook.


