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
encoder_name can be of *[Baseline, uniLSTM, BLSTM, BLSTMMax]*
*train.py* handles the downloading of the NLI Dataset and GloVe Embeddings. 

## Code Structure

- `models/`: Contains the implementation of the NLI models.
  - `AWE.py`: All models.
  - `NLINet.py`: NLInet. 
- `data_preprocessing.py`: Script for loading and preprocessing the dataset.
- `train.py`: Entry point for training models.
- `environment.yml`: List of packages required to run the models.
- `analysis.ipynb`: List of packages required to run the models.

## Link for Tensorboard logs and checkpoints
If you want to load the models directly you can download the checkpoints from the following link:
https://drive.google.com/drive/folders/1tCoCLzbVD8lxMtaAVzl2V0QG_Rnt-BA9
Note that you need to put this in the folder *checkpoints/*. After checkpoint loading you can play with the `analysis.ipynb` notebook, which explores the results.

## Additional Notes

- Make sure to adjust the batch size and other hyperparameters based on your system's capabilities.
- For detailed hyperparameter options and configurations, use the `--help` flag with the training or evaluation script.
- The cluster jobs are provides in the jobs folder

## Acknowledgements 

SNLI Dataset Citation:

> @article{bowman2015large,
>   title={A large annotated corpus for learning natural language inference},
>   author={Bowman, Samuel R and Angeli, Gabor and Potts, Christopher and Manning, Christopher D},
>   journal={arXiv preprint arXiv:1508.05326},
>   year={2015}
> }

Supervised Learning of Universal Sentence Representations from Natural Language Inference Data:

> @article{conneau2017supervised,
>   title={Supervised learning of universal sentence representations from natural language inference data},
>   author={Conneau, Alexis and Kiela, Douwe and Schwenk, Holger and Barrault, Loic and Bordes, Antoine},
>   journal={arXiv preprint arXiv:1705.02364},
>   year={2017}
> }




