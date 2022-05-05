import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from Trainer import Trainer
import os
import argparse
from datetime import datetime
from termcolor import colored
import random
import logging
import itertools
from LSTM_Model import RNN_Model, Dataset

logger = logging.getLogger(__name__)
ACTIVATIONS = {'relu': nn.ReLU, 'lrelu': nn.LeakyReLU, 'tanh': nn.Tanh}


def parsing():
    dt_string = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    parser = argparse.ArgumentParser()
    parser.add_argument('--tune_name', default="LSTMS")

    parser.add_argument('--wandb_mode', choices=['online', 'offline', 'disabled'], default='online', type=str)
    parser.add_argument('--project', default="Sepsis_Predictions", type=str)
    # parser.add_argument('--entity', default="surgical_data_science", type=str)
    parser.add_argument('--group', default=dt_string + " group ", type=str)
    # parser.add_argument('--use_gpu_num', default="0", type=str)

    parser.add_argument('--time_series_model', choices=['LSTM','GRU'], default='GRU', type=str)

    parser.add_argument('--activation', choices=['tanh','relu','lrelu'], default='tanh', type=str)
    parser.add_argument('--num_layers_rnn', default=4, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--bidirectional', default=False, type=bool)
    parser.add_argument('--dropout_rnn', default=0.4, type=float)

    parser.add_argument('--eval_rate', default=1, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--lr', default=0.05, type=float)
    parser.add_argument('--num_epochs', default=150, type=int)
    args = parser.parse_args()
    assert 0 <= args.dropout <= 1
    return args


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


    args = parsing()
    task_factor = [1] + 2 * [args.hands_factor]
    set_seed()
    logger.info(args)  # TODO : what is this?
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    experiment_name = args.group + " task:" + args.task_str
    logger.info(colored(experiment_name, "green"))

    # TODO - Load DataFrames for train, validation, test and create Datasets. Train model
    ds_train = Dataset()
    dl_train = DataLoader()
    ds_val = Dataset()
    dl_val = DataLoader()
    model = create_model(args)#TODO - Implement
    trainer = Trainer()

    #TODO: Calculate Validation and test F1 and Accuracy
