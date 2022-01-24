# Content-moderation

This project was developped to train a multi-level classification model for content moderation in images.

## Create Python Virtual Env

```bash
sudo apt-get update
sudo apt-get install python3.8
pip install virtualenv
virtualenv ohgodai
source ohgodai/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train the model

Clone and/or modify the config/training.yaml file then :

```bash
python train.py -i config/training.yaml
```

## Retrain from last checkpoint after interruption

```bash
python train.py -i config/training.yaml
```


## Retrain from specific checkpoint

In your config/training.yaml, the checkpoints will be at training/checkpoint_dir. You have a training_stats.tsv file with infos about the different checkpoints. You can choose one of the checkpoint and resume training (with different training parameters):

```bash
python train.py -i config/training.yaml -s <checkpoint_dir>/<specific-checkpoint>.pth
```
