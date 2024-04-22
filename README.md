# Sentence_representations_replication


## Quickstart Guide

### Replicating Results

1. Clone this repository:
```bash
git clone https://github.com/Jacksdab/Sentence_representations_replication.git
```
2. Create the environment:
```bash
conda env create -f environment.yml
```
3. Activate the environment:
```bash
conda activate atcs
```
4. Run the training of the models:
```bash
python train.py --encoder_name model
```
encoder_name can be of [Baseline, uniLSTM, BLSTM, BLSTMMax]


This will output the model's accuracy on the test set.

## Code Structure

- `models/`: Contains the implementation of the NLI models.
  - `AWE.py`: All models.
  - `NLINet.py`: NLInet. 
- `data_preprocessing.py`: Script for loading and preprocessing the dataset.
- `train.py`: Entry point for training models.
- `environment.yml`: List of packages required to run the models.
- `analysis.ipynb`: List of packages required to run the models.


## Additional Notes

- Make sure to adjust the batch size and other hyperparameters based on your system's capabilities.
- For detailed hyperparameter options and configurations, use the `--help` flag with the training or evaluation script.
- The cluster jobs are provides in the jobs folder



