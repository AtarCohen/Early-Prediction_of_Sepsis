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
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer,SimpleImputer

#%%


logger = logging.getLogger(__name__)
Imputations = {}
FREQUENCY_ATTR =['5w_sum_BaseExcess', '5w_sum_FiO2', '5w_sum_pH', '5w_sum_PaCO2', '5w_sum_Glucose', '5w_sum_Lactate', '5w_sum_PTT']
LAB_ATTR = [ 'Hct',  'Glucose','Potassium']
CONST_ATTR = ['ID','max_ICULOS','Gender']
OTHER_ATTR = ['HR','MAP','O2Sat', 'Resp','SBP','ICULOS']
LABEL_ATTR= 'Label'
COLS = FREQUENCY_ATTR+CONST_ATTR+LAB_ATTR+OTHER_ATTR
TRAIN_PATH = '/home/student/filtered_train_df_0705_LSTM_new.csv'
VAL_PATH = '/home/student/filtered_val_df_0705_LSTM_new.csv'
TEST_PATH = '/home/student/filtered_test_df_0705_LSTM_new.csv'



def parsing():
    dt_string = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    parser = argparse.ArgumentParser()
    parser.add_argument('--tune_name', default="LSTMS")

    parser.add_argument('--wandb_mode', choices=['online', 'offline', 'disabled'], default='online', type=str)
    parser.add_argument('--project', default="Sepsis_Predictions", type=str)
    # parser.add_argument('--entity', default="surgical_data_science", type=str)
    parser.add_argument('--group', default=dt_string + " group ", type=str)
    # parser.add_argument('--use_gpu_num', default="0", type=str)

    parser.add_argument('--time_series_model', choices=['LSTM','GRU'], default='LSTM', type=str)

    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--bidirectional', default=False, type=bool)
    parser.add_argument('--dropout', default=0.4, type=float)

    parser.add_argument('--eval_rate', default=1, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--num_epochs', default=100, type=int)

    # parser.add_argument('--imputation', choices=['iterative','mean','median'], default='iterative', type=str)
    parser.add_argument('--window', default=5, type=int)
    parser.add_argument('--seq_len', default=5, type=int)
    parser.add_argument('--sample', choices=['over','under','overunder'], default='overunder', type=str)
    parser.add_argument('--over_sample_rate', default='0.3', type=float)
    parser.add_argument('--under_sample_rate', default='0.5', type=float)

    args = parser.parse_args()
    assert 0 <= args.dropout <= 1
    return args


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# def prepare_data(df,args,type):
#     df = df[df['time_bm']>=-1*(args.seq_len+args.window)]
#     df = add_rolling_window(df, FREQUENCY_ATTR, args.window)
#     df = df[df['time_bm']>=-1*(args.seq_len)]
#     if type=='train':
#         patient_ids = sample(df, args)
#     else:
#         patient_ids = list(set(df.ID.values))
#     df= df[COLS]
#     df = impute_per_patient(df,args)
#     return df, patient_ids
#
#
# class data_preperator():
#     def __init__(self,train_df,seq_len=10,window=5):
#         self.all_data_means= train_df.mean()
#         self.seq_len=seq_len
#         self.window = window
#
#     def impute_per_patient(self,df):
#         patients = list(set(df.ID.values))
#         imputed = pd.DataFrame()
#         for patient in patients:
#             tmp_df = df[df['ID']==patient][COLS+['Label']]
#             for f in LAB_ATTR+OTHER_ATTR:
#                 if tmp_df[f].isnull().all:
#                     tmp_df[f]=tmp_df[f].fillna(self.all_data_means[f])
#             imp = IterativeImputer(max_iter=50, random_state=0)
#             try:
#                 imp.fit(tmp_df)
#                 tmp_df= pd.DataFrame(imp.transform(tmp_df), columns = COLS+['Label'])
#                 imputed=imputed.append(tmp_df)
#             except:
#                 print(tmp_df.shape)
#         return imputed
#
#     def add_rolling_window(self,df, attr, window_size):
#         df = df.sort_values(by=['ID','ICULOS'], ascending =[True,True])
#         rolling = df[['ID']+attr].groupby('ID').rolling(window=window_size, closed='both').count()
#         rolling= rolling.rename(columns={at: f'{window_size}w_sum_{at}' for at in attr})
#         rolling=rolling[list(rolling.columns)[1:]].reset_index().set_index('level_1')
#         combined = df.join(rolling,how='left', rsuffix= 'r')
#         return combined
#
#     def prepare_data(self,df):
#         df = df[df['time_bm']>=-1*(self.seq_len+self.window)]
#         df = self.add_rolling_window(df, frequency_used_attributes, self.window)
#         df = df[df['time_bm']>=-1*(self.seq_len)]
#         df= df[COLS+['Label']]
#         df = self.impute_per_patient(df)
#         return df



def sample(df,args):
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
cols = list(train_df.columns)
cols.remove('Label')
cols.remove('ID')
train_patients= sample(train_df, args)
input_dim = len(cols)
ds_train = Dataset(train_patients,train_df,cols)
dl_train = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_inputs, shuffle=True)
val_df = pd.read_csv(VAL_PATH)
val_patients= list(set(val_df.ID.values))
ds_val = Dataset(val_patients,val_df,cols)
dl_val = DataLoader(ds_val, batch_size=args.batch_size, collate_fn=collate_inputs)
model = RNN_Model(rnn_type=args.time_series_model,bidirectional=args.bidirectional,input_dim = input_dim,hidden_dim = args.hidden_dim,dropout= args.dropout,num_layers =args.num_layers )
trainer = Trainer(model)
eval_results, train_results = trainer.train(dl_train,dl_val,args.num_epochs, args.lr,args)
