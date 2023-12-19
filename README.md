# EC-523 Project: Enhancing the SCRUB Model

## Introduction
This repository contains the re-implementation and enhancement of the SCalable Remembering and Unlearning unBound (SCRUB) model. The SCRUB model represents a novel approach in [briefly describe the main function or advantage of the model]. This project aims to [state the objectives of your enhancements or what aspects of the SCRUB model you focused on].

## Prerequisites
To install the prerequsites librarys run
```shell
pip install -r requirements.txt
```

## Usage
### Obtaining the Models
To download and set up the pre-trained and re-trained SCRUB models in your local environment, run:
```python3
python3 train_checkpoints.py
```
## Performing Unlearning
After obtaining the pre-trained model, use the following commands to perform unlearning:

SCRUB Method:
```
python3 main.py --checkpoints --sgda_learning_rate 0.006 --sub_sample 0.0 --unlearning_method "SCRUB" 
```

Finetuning Method:
```
python3 main.py --checkpoints --sgda_learning_rate 0.1 --unlearning_method "finetuning" 
```

Negative Gradient Method:

```
python3 main.py --checkpoints --sgda_learning_rate 0.006 --unlearning_method "negative" --sgda_epochs 5
```

## Hyperparameter Tuning
Use the following scripts to tune hyperparameters:

```run_lr.sh ```\
```run_subsampling.sh ```\
```run_temp.sh ```

## Reviewing Results
All results can be reviewed in the checkpoints.ipynb notebook.