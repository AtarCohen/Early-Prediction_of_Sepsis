import sys
import os
import joblib
import pandas as pd
from sklearn.metrics import f1_score
import tqdm
import argparse
import pickle
from DataPreaparators import PreProcess, create_patients_df
import random

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
#

#
# class PreProcess():
#     """
#     This Class is used to prepare the data for the non-time series models
#     We add 4 types of columns: window columns and frequency columns,SOFA column and unit3 column, as explained in the
#     report. and then aggregate the results to get 1 record (row in the dataframe) for each patient
#     """
#     def __init__(self,df,window_size = 5, time_bm=-10, imputer_path = None):
#         """
#         :param df: data frame to process
#         :param window_size: size of the window to calculate number of tests
#         :param time_bm: maximum time before sepsis to filter the data frame
#         :param imputer_path: path to trained imputer
#         """
#         self.frequency_used_attributes = ['BaseExcess', 'FiO2', 'pH', 'PaCO2', 'Glucose', 'Lactate', 'PTT']
#         self.freq_columns = ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2',
#                  'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride',
#                  'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate',
#                  'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total',
#                  'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets']
#         self.freq_columns_final = [f'freq_{at}' for at in self.freq_columns]
#         self.values_used_attributes = ['Hct', 'Glucose', 'Potassium']
#         self.constant_attributes = ['ID', 'max_ICULOS', 'Gender']
#         self.other_attributes = ['time_bm', 'HR', 'MAP', 'O2Sat', 'Resp', 'SBP', 'ICULOS']
#         self.units_attributes = ['Unit1', 'Unit2']
#         self.label_attributes = ['Label', 'SepsisLabel']
#         self.window_size = window_size
#         self.time_bm = time_bm
#         self.test_df = self.preprocess(df)
#         self.IDS = self.test_df[['ID']]
#         self.knn_imp = joblib.load(imputer_path)
#         self.cols = list(self.test_df.columns)
#         self.cols.remove('ID')
#         self.cols.remove('Label')
#         # with open(f'Best_features_RF_run.pickle', 'rb') as handle:
#         #     self.features = pickle.load(handle)
#         self.features = ['max_ICULOS', 'freq_pH', 'freq_TroponinI', 'freq_Bilirubin_direct', 'AST__mean', 'Glucose__mean', 'MAP__min', 'freq_Chloride', 'WBC__median', 'freq_BUN', 'O2Sat__std', 'pH__std', 'WBC__min', 'freq_PaCO2', 'freq_Creatinine', 'PTT__mean', 'Hct__min', 'Glucose__std', 'EtCO2__mean', 'freq_FiO2', 'freq_Glucose', 'freq_Fibrinogen', 'Hgb__mean', 'SaO2__mean', 'freq_SaO2', 'O2Sat__mean', 'Platelets__mean']
#         self.X = pd.DataFrame(self.knn_imp.transform(self.test_df[self.cols]), columns=self.cols)[self.features]
#         # self.X_val = self.val_df.fillna(0)[self.columns]
#         self.y = self.test_df['Label'].values
#
#     # create frequency columns for some lab variables
#     def add_rolling_window(self, df):
#         df = df.sort_values(by=['ID', 'ICULOS'], ascending=[True, True])
#         rolling = df[['ID'] + self.frequency_used_attributes].groupby('ID').rolling(window=self.window_size,
#                                                                                     closed='both').count()
#         rolling = rolling.rename(columns={at: f'{self.window_size}w_sum_{at}' for at in self.frequency_used_attributes})
#         rolling = rolling[list(rolling.columns)[1:]].reset_index().set_index('level_1')
#         combined = df.join(rolling, how='left', rsuffix='r')
#         return combined, rolling
#
#     def calc_frequency(self, df):
#         df = df.sort_values(by=['ID', 'ICULOS'], ascending=[True, True])
#         rolling = df[['ID', 'ICULOS'] + self.freq_columns].groupby(by=['ID'])[
#             self.freq_columns].expanding().count().reset_index().rename(columns={'level_1': 'old_index'})
#         df = df.reset_index().rename(columns={'index': 'old_index'})
#         rolling = rolling.rename(columns={at: f'freq_{at}' for at in self.freq_columns})
#         combined = pd.merge(df, rolling, on=['ID', 'old_index'])
#         new_cols = [f'freq_{at}' for at in self.freq_columns]
#         for at in new_cols:
#             combined[at] = combined[at] / combined['ICULOS']
#         freq_df = combined[['ID'] + self.freq_columns_final].groupby(by='ID').last().reset_index()
#         return freq_df
#
#     def preprocess(self,df):
#         df, df_roll = self.add_rolling_window(df)
#         freq_df = self.calc_frequency(df)
#         frequency_used_attributes_fixed = [f'{self.window_size}w_sum_{x}' for x in self.frequency_used_attributes]
#         df = df[df['time_bm']>=self.time_bm]
#         # handle Units123
#         df['Unit3'] = ( (1*(df['Unit1']+df['Unit2'])<1) |
#                               (df['Unit1'].isna() & df['Unit2'].isna()) )*1
#         df['Unit1'][df['Unit1'].isna()] = 0
#         df['Unit2'][df['Unit2'].isna()] = 0
#
#         # create SOFA attribute
#         df['SOFA'] = df['SBP'] <= 100
#         df['SOFA'] += df['Resp'] >= 22
#
#         # aggregations
#
#         data_final = df.groupby(['ID', 'Label','max_ICULOS','Gender']).agg({
#                                                             'SOFA': 'max', \
#                                                             'Unit1': 'max',\
#                                                             'Unit2': 'max',\
#                                                             'Unit3': 'max',\
#                                                             'HR': ['median', 'max','std'],\
#                                                             'MAP': ['median', 'min'],\
#                                                             'O2Sat': ['mean','std'],\
#                                                             'Resp': ['median', 'max','std'],\
#                                                             'SBP': ['median', 'min','std'],\
#                                                             'Hct': ['median', 'min'],\
#                                                             'Potassium': 'mean',\
#                                                             'Glucose': ['mean','std'],\
#                                                             'Temp': ['mean', 'min'],\
#                                                             'DBP': 'mean',\
#                                                             'WBC': ['median', 'min','std'],\
#                                                             'EtCO2': 'mean',\
#                                                             'BaseExcess': 'mean',\
#                                                             'HCO3': 'mean',\
#                                                             'FiO2': ['mean','std'],\
#                                                             'SaO2': 'mean',\
#                                                             'AST': 'mean',\
#                                                             'Lactate': 'mean',\
#                                                             'Magnesium': 'mean',\
#                                                             'Phosphate': 'mean',\
#                                                             'TroponinI': 'mean',\
#                                                             'Hgb': 'mean',\
#                                                             'PTT': 'mean',\
#                                                             'Platelets': 'mean',\
#                                                             'Age': 'mean',\
#                                                             'HospAdmTime': 'mean',\
#                                                             'pH': ['std','median'],\
#                                                             f'{self.window_size}w_sum_BaseExcess': 'mean',\
#                                                             f'{self.window_size}w_sum_FiO2': 'mean',\
#                                                             f'{self.window_size}w_sum_pH': 'mean',\
#                                                             f'{self.window_size}w_sum_PaCO2': 'mean',\
#                                                             f'{self.window_size}w_sum_Glucose': 'mean',\
#                                                             f'{self.window_size}w_sum_Lactate': 'mean',\
#                                                             f'{self.window_size}w_sum_PTT': 'mean'}).reset_index()
#         data_final.columns = ['__'.join(col).strip() for col in data_final.columns.values]
#         data_final.rename(columns={"ID__": "ID", "Label__": "Label", "max_ICULOS__":"max_ICULOS", "Gender__":"Gender"}, inplace=True)
#         data_final['SOFA__max'] += data_final['SBP__median'] <= 100
#         data_final['SOFA__max'] += data_final['Resp__median'] >= 22
#         data_final = pd.merge(data_final,freq_df, on='ID')
#         return data_final


if __name__ == "__main__":
    # test_files= sys.argv[1]
    # patients = os.listdir(test_files)
    # test_df = create_patients_df(patients,test_files)
    test_df = pd.read_csv('/home/student/filtered_test_df_0705.csv')
    preprocessor= PreProcess(df=test_df, imputer_path='knn_imputer')
    model = joblib.load('best_XGB_run3_43')
    with open(f'Best_features_XGB_run_4.pickle', 'rb') as handle:
        features = pickle.load(handle)
    random.seed(0)
    y_pred = model.predict(preprocessor.X[features])
    print('F1 Score: ',f1_score(preprocessor.y,y_pred))
    final_results =preprocessor.IDS
    final_results['SepsisLabel'] = y_pred
    # final_results.to_csv('results.csv', index=False)