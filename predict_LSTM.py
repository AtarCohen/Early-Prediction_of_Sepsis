import sys
import os
import joblib
import pandas as pd
from sklearn.metrics import f1_score
import tqdm
from torch.utils.data import DataLoader
from LSTM_Model import RNN_Model, Dataset, collate_inputs
import torch
import argparse
import random

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer,SimpleImputer
from DataPreaparators import DataPreparator, create_patients_df
#
# def create_patients_df(patients, data_path):
#     """
#     :param patients: all file names in the given directory
#     :param data_path: directory path
#     :return: Patients DataFrame with all the data Provided for each patient
#                                             + 2 new columns: max_ICULOS and time_bm=ICULOS-time_bm
#     """
#     tmp_df = pd.read_csv(os.path.join(data_path,patients[0]),delimiter ='|')
#     tmp_df['ID'] = patients[0].split('_')[-1].split('.')[0]
#     new_df = tmp_df[tmp_df['SepsisLabel']==0]
#     if max(tmp_df['SepsisLabel'])==1:
#         new_df=new_df.append(tmp_df[tmp_df['SepsisLabel']==1][:1])
#         new_df['Label'] = [1]*new_df.shape[0]
#     else:
#         new_df['Label'] = [0]*new_df.shape[0]
#     new_df['max_ICULOS'] = [new_df['ICULOS'].values[-1]]*new_df.shape[0]
#     new_df['time_bm'] =  new_df['ICULOS']-new_df['max_ICULOS']
#     for patient in tqdm.tqdm(patients[1:]):
#         patient_path = os.path.join(data_path,patient)
#         patient_number = patient.split('_')[-1].split('.')[0]
#         tmp_df = pd.read_csv(os.path.join(data_path,patient_path),delimiter ='|')
#         tmp_df['ID'] = patient_number
#         tmp_new_df = tmp_df[tmp_df['SepsisLabel']==0]
#         if max(tmp_df['SepsisLabel'])==1:
#             tmp_new_df=tmp_new_df.append(tmp_df[tmp_df['SepsisLabel']==1][:1])
#             tmp_new_df['Label'] = [1]*tmp_new_df.shape[0]
#         else:
#             tmp_new_df['Label'] = [0]*tmp_new_df.shape[0]
#         tmp_new_df['max_ICULOS'] = [tmp_new_df['ICULOS'].values[-1]]*tmp_new_df.shape[0]
#         tmp_new_df['time_bm'] =  tmp_new_df['ICULOS']-tmp_new_df['max_ICULOS']
#         new_df = new_df.append(tmp_new_df)
#     return new_df
# #
# #
# class DataPreparator():
#     """
#     This Class will add columns and impute the data for each patient
#     """
#     def __init__(self,columns,window_columns=None, freq_columns=None, seq_len=10,window=5):
#         self.all_data_means= pd.read_csv('/home/student/filtered_train_mean.csv') #mean values for all possibly null atributes
#         self.seq_len=seq_len
#         self.window = window
#         self.window_columns = window_columns
#         self.freq_columns = freq_columns
#         self.freq_columns_final =  [f'freq_{at}' for at in self.freq_columns]
#         self.columns = columns
#
#     def impute_per_patient(self,df):
#         """
#         :param df: data frame to impute
#         :return: imputed data
#         """
#         patients = list(set(df.ID.values))
#         imputed = pd.DataFrame()
#         for patient in patients:
#             tmp_df = df[df['ID']==patient][self.columns+self.freq_columns_final+['time_bm']]
#             # tmp_labels = df[df['ID']==patient]['Label']
#             for f in self.columns:
#                 if tmp_df[f].isnull().all:
#                     mean_val = self.all_data_means[self.all_data_means['index']==f]['0'].values[0]
#                     tmp_df[f]=tmp_df[f].fillna(mean_val)
#             imp = IterativeImputer(max_iter=50, random_state=0)
#             # try:
#             imp.fit(tmp_df)
#             tmp_df= pd.DataFrame(imp.transform(tmp_df), columns = self.columns+self.freq_columns_final+['time_bm'])
#             tmp_df['Label'] = df[df['ID']==patient]['Label'].values
#             tmp_df['ID'] = [patient]*tmp_df.shape[0]
#             imputed=imputed.append(tmp_df)
#         return imputed
#
#     def add_rolling_window(self,df):
#         df = df.sort_values(by=['ID','ICULOS'], ascending =[True,True])
#         rolling = df[['ID']+self.window_columns].groupby('ID').rolling(window=self.window, closed='both').count()
#         rolling= rolling.rename(columns={at: f'{self.window}w_sum_{at}' for at in self.window_columns})
#         rolling=rolling[list(rolling.columns)[1:]].reset_index().set_index('level_1')
#         combined = df.join(rolling,how='left', rsuffix= 'r')
#         self.columns +=[f'{self.window}w_sum_{at}' for at in self.window_columns]
#         return combined
#
#     def add_frequency(self,df):
#         df = df.sort_values(by=['ID', 'ICULOS'], ascending=[True, True])
#         rolling = df[['ID','ICULOS']+self.freq_columns].groupby(by=['ID'])[self.freq_columns].expanding().count().reset_index().rename(columns={'level_1':'old_index'})
#         df=df.reset_index().rename(columns={'index':'old_index'})
#         rolling = rolling.rename(columns={at: f'freq_{at}' for at in self.freq_columns})
#         combined = pd.merge(df,rolling, on=['ID','old_index'])
#         for at in self.freq_columns_final:
#             combined[at] = combined[at] / combined['ICULOS']
#         return combined
#
#
#     def prepare_data(self,df, rolling=False,freq=True):
#         if rolling:
#             df = self.add_rolling_window(df)
#         if freq:
#             df = self.add_frequency(df)
#         df = df[df['time_bm']>=-1*(self.seq_len)]
#         df = df[self.columns+self.freq_columns_final+['time_bm','ID','Label']]
#         df = self.impute_per_patient(df)
#         return df
# #%%


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def predict_and_eval(model,test_data_loader):
    model.eval()
    ids= []
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_data_loader:
            batch_input, batch_target, lengths, mask, batch_ids = batch
            lengths = lengths.to(dtype=torch.int64).to(device='cpu')
            predictions = model(batch_input, lengths, mask)
            _, predicted = torch.max(predictions, 1)
            all_preds += predicted
            all_labels += batch_target
            ids += batch_ids
        f1 = f1_score(all_labels,all_preds)
        print(f'F1 Score: {f1}')
    results = pd.DataFrame({'ID':ids,'SepsisLabel':all_preds})
    return results



if __name__ == "__main__":
    frequency_used_attributes = ['BaseExcess', 'FiO2', 'pH', 'PaCO2', 'Glucose', 'Lactate', 'PTT']
    # FREQUENCY_ATTR =['5w_sum_BaseExcess', '5w_sum_FiO2', '5w_sum_pH', '5w_sum_PaCO2', '5w_sum_Glucose', '5w_sum_Lactate', '5w_sum_PTT']
    # LAB_ATTR = ['Hct',  'Glucose','Potassium']
    CONST_ATTR = ['max_ICULOS', 'Gender']
    OTHER_ATTR = ['HR', 'MAP', 'O2Sat', 'Resp', 'SBP', 'ICULOS']
    ALL_LAB_ATTR = ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2',
                    'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride',
                    'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate',
                    'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total',
                    'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets']
    COLS = CONST_ATTR + OTHER_ATTR

    # test_files= sys.argv[1]
    # patients = os.listdir(test_files)
    # test_df = create_patients_df(patients,test_files)
    # p = DataPreparator(columns=COLS, freq_columns=ALL_LAB_ATTR)
    # test_df = p.prepare_data(test_df)
    test_df = pd.read_csv('/home/student/filtered_test_df_0705_LSTM_new.csv')
    test_patients = list(set(test_df.ID.values))
    cols = list(test_df.columns)
    cols.remove('Label')
    cols.remove('ID')
    ds = Dataset(test_patients, test_df, cols)
    set_seed()
    dl = DataLoader(ds, batch_size=64, collate_fn=collate_inputs)
    model = RNN_Model(rnn_type='GRU', bidirectional=False, input_dim=35,
                      hidden_dim=256, dropout=0.3922844934594849 , num_layers=3)
    model.load_state_dict(torch.load('Trained Models/184.pth')['model_state'])
    final_results = predict_and_eval(model,dl)
    # final_results.to_csv('results_LSTM.csv', index=False)




