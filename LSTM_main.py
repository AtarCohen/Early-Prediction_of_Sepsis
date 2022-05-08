import torch
from torch import nn
from torch.utils.data import DataLoader
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
from LSTM_Model import RNN_Model, Dataset, collate_inputs

logger = logging.getLogger(__name__)
Imputations = {}
FREQUENCY_ATTR = [ f'{5}w_sum_{at}' for at in ['BaseExcess',  'FiO2', 'pH', 'PaCO2', 'Glucose','Lactate', 'PTT']]
LAB_ATTR = [ 'Hct',  'Glucose','Potassium']
CONST_ATTR = ['ID','max_ICULOS','Gender']
OTHER_ATTR = ['time_bm','HR','MAP','O2Sat', 'Resp','SBP','ICULOS']
LABEL_ATTR= 'Label'
COLS = FREQUENCY_ATTR+LAB_ATTR+OTHER_ATTR
TRAIN_PATH = 'filtered_train_df_0705.csv'
VAL_PATH = 'filtered_val_df_0705.csv'
TEST_PATH = 'filtered_test_df_0705.csv'



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

    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--bidirectional', default=False, type=bool)
    parser.add_argument('--dropout', default=0.4, type=float)

    parser.add_argument('--eval_rate', default=1, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--lr', default=0.05, type=float)
    parser.add_argument('--num_epochs', default=1, type=int)

    parser.add_argument('--imputation', choices=['iterative','mean','median'], default='iterative', type=str)
    parser.add_argument('--window', default=5, type=int)
    parser.add_argument('--seq_len', default=10, type=int)
    parser.add_argument('--sampling', choices=['over','under'], default='iterative', type=str)

    args = parser.parse_args()
    assert 0 <= args.dropout <= 1
    return args


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def add_rolling_window(df, attr, window_size):
    df = df.sort_values(by=['ID','ICULOS'], ascending =[True,True])
    rolling = df[['ID']+attr].groupby('ID').rolling(window=window_size, closed='both').count()
    rolling= rolling.rename(columns={at: f'{window_size}w_sum_{at}' for at in attr})
    rolling=rolling[list(rolling.columns)[1:]].reset_index().set_index('level_1')
    combined = df.join(rolling,how='left', rsuffix= 'r')
    return combined


def prepare_data(df,args):
    df = df[df['time_bm']>=-1*(args.seq_len+args.window)]
    df = add_rolling_window(df, FREQUENCY_ATTR, args.window)
    df = df[df['time_bm']>=-1*(args.seq_len)]
    return df

def impute_per_patient(df,args):


def impute_and_sample(df,args):
    if args.sampling=='under':






args = parsing()
set_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
full_df = pd.read_csv('/home/student/filtered_train_data_combined.csv')
max_los = full_df[['ID','ICULOS']].groupby(by='ID').max().rename(columns={'ICULOS':'Max_ICULOS'})
full_df=full_df.join(max_los,on='ID',how='left')
full_df['time_bm'] =  full_df['ICULOS']-full_df['Max_ICULOS']
full_df = full_df[full_df['time_bm']>=-10]
full_df = full_df.fillna(-1)
# TODO - Load DataFrames for train, validation, test and create Datasets. Train model
ds_train = Dataset(full_df[(full_df['ID']==1)|(full_df['ID']==2)][:-2])
dl_train = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_inputs, shuffle=True)
ds_val = Dataset(full_df[(full_df['ID']==1)|(full_df['ID']==2)][:-2])
dl_val = DataLoader(ds_val, batch_size=args.batch_size, collate_fn=collate_inputs)
model = RNN_Model(rnn_type=args.time_series_model,input_dim = INPUT_DIM,hidden_dim = args.hidden_dim,dropout= args.dropout,num_layers =args.num_layers )
trainer = Trainer(model)
eval_results, train_results = trainer.train(dl_train,dl_val,args.num_epochs, args.lr,args)


