from ast import arg
from OLOFT.data_utils import get_autogluon_datasets
import argparse
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from tqdm.autonotebook  import tqdm
import random
import torch
import os 
from autogluon.multimodal import MultiModalPredictor

args = argparse.ArgumentParser()
args.add_argument('--seed', type=int, default=0, help='Random seed used to make the training repeatable. Default: 0')
args.add_argument('--dataset_folder', type=str, default='./data/processed', help='Path to the folder including the preprocessed data including images generate. Default: ./data/processed')
args.add_argument('--model_folder', type=str, default='./models', help='Path to the folder to save the trained model. Default: ./models')
args = args.parse_args()

# if the model folder existis remove it
if os.path.exists(args.model_folder):
    os.system(f'rm -rf {args.model_folder}')

# set the random seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# get the datasets
train_data, val_data, test_data = get_autogluon_datasets(target_folder=args.dataset_folder)

# initialize the multimodal predictor
predictor = MultiModalPredictor(label="label", path=args.model_folder)
predictor.fit(
    train_data=train_data,
    tuning_data=val_data,
    time_limit=18000,
    seed = args.seed
)