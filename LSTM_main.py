import torch
from torch import nn
import numpy as np
from trainer import Trainer
import os
import argparse
from datetime import datetime
# from termcolor import colored
import random
import logging
import itertools
import pandas as pd
from torch.utils.data import DataLoader
from LSTM_Model import RNN_Model, Dataset, collate_inputs
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer,SimpleImputer

#%%


logger = logging.getLogger(__name__)
Imputations = {}
TRAIN_PATH = 'DataFiles/filtered_train_df_0705_LSTM_new.csv'
VAL_PATH = 'DataFiles/filtered_val_df_0705_LSTM_new.csv'
TEST_PATH = 'DataFiles/filtered_test_df_0705_LSTM_new.csv'



def parsing():
    dt_string = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    parser = argparse.ArgumentParser()
    parser.add_argument('--tune_name', default="LSTMS")

    parser.add_argument('--wandb_mode', choices=['online', 'offline', 'disabled'], default='online', type=str)
    parser.add_argument('--project', default="Sepsis_Predictions", type=str)
    # parser.add_argument('--entity', default="surgical_data_science", type=str)

    parser.add_argument('--time_series_model', choices=['LSTM','GRU'], default='GRU', type=str)

    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--hidden_dim', default=32, type=int)
    parser.add_argument('--bidirectional', default=False, type=bool)
    parser.add_argument('--dropout', default=0.4526, type=float)

    parser.add_argument('--eval_rate', default=1, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=0.001497, type=float)
    parser.add_argument('--num_epochs', default=100, type=int)

    # parser.add_argument('--imputation', choices=['iterative','mean','median'], default='iterative', type=str)
    parser.add_argument('--window', default=5, type=int)
    parser.add_argument('--seq_len', default=10, type=int)
    parser.add_argument('--sample', choices=['over','under','overunder'], default='overunder', type=str)
    parser.add_argument('--over_sample_rate', default='0.3449', type=float)
    parser.add_argument('--under_sample_rate', default='0.5034', type=float)

    args = parser.parse_args()
    assert 0 <= args.dropout <= 1
    return args


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True



def sample(df,args):
    """
    Over and Under sample from the train dataset
    :param df:  train dataframe
    :param args:  include sample rates
    :return: list of sampled ids (oversampled from minority class and under sampled from majority class)
    """
    df = df[['ID', 'Label']].groupby(by='ID').max().reset_index()
    ids = df[['ID']]
    labels = df['Label']
    if 'over' in args.sample:
        over = RandomOverSampler(sampling_strategy=args.over_sample_rate)
        ids,labels = over.fit_resample(ids,labels)
    if 'under' in args.sample:
        under = RandomUnderSampler(sampling_strategy=args.under_sample_rate)
        ids,labels = under.fit_resample(ids,labels)
    return ids['ID'].values


args = parsing()
set_seed()
logger.info(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_df = pd.read_csv(TRAIN_PATH)
cols = list(train_df.columns) #input for the model
cols.remove('Label') #this is not an input
cols.remove('ID')
train_patients= sample(train_df, args)
input_dim = len(cols)
ds_train = Dataset(train_patients,train_df,cols)
dl_train = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_inputs, shuffle=True)
val_df = pd.read_csv(VAL_PATH)
val_patients= list(set(val_df.ID.values))
ds_val = Dataset(val_patients,val_df,cols)
dl_val = DataLoader(ds_val, batch_size=args.batch_size, collate_fn=collate_inputs, shuffle=False)
model = RNN_Model(rnn_type=args.time_series_model,bidirectional=args.bidirectional,input_dim = input_dim,hidden_dim = args.hidden_dim,dropout= args.dropout,num_layers =args.num_layers )
trainer = Trainer(model)
eval_results, train_results = trainer.train(dl_train,dl_val,args.num_epochs, args.lr,args)

# test_df = pd.read_csv(TEST_PATH)
# test_patients= list(set(test_df.ID.values))
# ds_test = Dataset(test_patients,test_df,cols)
# dl_test = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_inputs)
# print(trainer.eval(dl_test,name='test'))
