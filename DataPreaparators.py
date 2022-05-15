import pandas as pd
import random
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer,SimpleImputer
from sklearn.impute import KNNImputer
import os
import tqdm
import joblib

"""
Includes helper function for data preprocessing:
1. Reading patients data from given directory
2. Class to preprocess data for RNN models
3. class to preprocess data for non-temporal models

"""
def create_patients_df(patients, data_path):
    """
    :param patients: all file names in the given directory
    :param data_path: directory path
    :return: Patients DataFrame with all the data Provided for each patient
                                            + 2 new columns: max_ICULOS and time_bm=ICULOS-time_bm
                                            For each patient we add ID from the file name
    """
    tmp_df = pd.read_csv(os.path.join(data_path,patients[0]),delimiter ='|') #reading first file
    tmp_df['ID'] = patients[0].split('_')[-1].split('.')[0]
    new_df = tmp_df[tmp_df['SepsisLabel']==0] #all SepsisLabel=0 rows
    if max(tmp_df['SepsisLabel'])==1: #if patient had sepsis
        new_df=new_df.append(tmp_df[tmp_df['SepsisLabel']==1][:1]) #add first row where SepsisLabel=1
        new_df['Label'] = [1]*new_df.shape[0]
    else:
        new_df['Label'] = [0]*new_df.shape[0]
    new_df['max_ICULOS'] = [new_df['ICULOS'].values[-1]]*new_df.shape[0] #calculate max_iculos
    new_df['time_bm'] =  new_df['ICULOS']-new_df['max_ICULOS']
    for patient in tqdm.tqdm(patients[1:]): ##repeat the process for the rest of the patient in the directory
        patient_path = os.path.join(data_path,patient)
        patient_number = patient.split('_')[-1].split('.')[0]
        tmp_df = pd.read_csv(os.path.join(data_path,patient_path),delimiter ='|')
        tmp_df['ID'] = patient_number
        tmp_new_df = tmp_df[tmp_df['SepsisLabel']==0]
        if max(tmp_df['SepsisLabel'])==1:
            tmp_new_df=tmp_new_df.append(tmp_df[tmp_df['SepsisLabel']==1][:1])
            tmp_new_df['Label'] = [1]*tmp_new_df.shape[0]
        else:
            tmp_new_df['Label'] = [0]*tmp_new_df.shape[0]
        tmp_new_df['max_ICULOS'] = [tmp_new_df['ICULOS'].values[-1]]*tmp_new_df.shape[0]
        tmp_new_df['time_bm'] =  tmp_new_df['ICULOS']-tmp_new_df['max_ICULOS']
        new_df = new_df.append(tmp_new_df) #append to one dataframe
    return new_df




class DataPreparator():
    """
    This Class will add columns and impute the data for each patient, preparing it to RNN models
    """
    def __init__(self,columns,window_columns=None, freq_columns=None, seq_len=10,window=5):
        """
        :param columns: columns from the original df to return for the model
        :param window_columns: columns from the original df to calculate "windows" - sum of non-null values in some interval
        :param freq_columns: columns from original df to calculate frequency - rate of non-null values along the length
        stay in the ICU
        :param seq_len: number of rows to save for each patient (last <seq_len> hours logged in the ICU)
        :param window: interval size for window calculation
        """
        self.all_data_means= pd.read_csv('/home/student/filtered_train_mean.csv') #mean values for all possibly null atributes
        self.seq_len=seq_len
        self.window = window
        self.window_columns = window_columns
        self.freq_columns = freq_columns
        self.freq_columns_final =  [f'freq_{at}' for at in self.freq_columns]
        self.columns = columns

    def impute_per_patient(self,df):
        """
        :param df: data frame to impute
        :return: imputed data
        """
        patients = list(set(df.ID.values))
        imputed = pd.DataFrame()
        for patient in patients:
            tmp_df = df[df['ID']==patient][self.columns+self.freq_columns_final+['time_bm']]
            # tmp_labels = df[df['ID']==patient]['Label']
            for f in self.columns:
                if tmp_df[f].isnull().all:
                    mean_val = self.all_data_means[self.all_data_means['index']==f]['0'].values[0]
                    tmp_df[f]=tmp_df[f].fillna(mean_val)
            imp = IterativeImputer(max_iter=50, random_state=0)
            # try:
            imp.fit(tmp_df)
            tmp_df= pd.DataFrame(imp.transform(tmp_df), columns = self.columns+self.freq_columns_final+['time_bm'])
            tmp_df['Label'] = df[df['ID']==patient]['Label'].values
            tmp_df['ID'] = [patient]*tmp_df.shape[0]
            imputed=imputed.append(tmp_df)
        return imputed

    def add_rolling_window(self,df):
        """
        calculates "window" columns - For wach window_column, we calculate the number of non-null values in the
        self.window (which is an int ) size.
        :param df: datafrane to process
        :return: dataframe with "window" columns added
        """
        df = df.sort_values(by=['ID','ICULOS'], ascending =[True,True])
        rolling = df[['ID']+self.window_columns].groupby('ID').rolling(window=self.window, closed='both').count()
        rolling= rolling.rename(columns={at: f'{self.window}w_sum_{at}' for at in self.window_columns})
        rolling=rolling[list(rolling.columns)[1:]].reset_index().set_index('level_1')
        combined = df.join(rolling,how='left', rsuffix= 'r')
        self.columns +=[f'{self.window}w_sum_{at}' for at in self.window_columns]
        return combined

    def add_frequency(self,df):
        """
        :param df:
        :return: dataframe with frequency columns for all self.freq_columns
        """
        df = df.sort_values(by=['ID', 'ICULOS'], ascending=[True, True])
        rolling = df[['ID','ICULOS']+self.freq_columns].groupby(by=['ID'])[self.freq_columns].expanding().count().reset_index().rename(columns={'level_1':'old_index'})
        df=df.reset_index().rename(columns={'index':'old_index'})
        rolling = rolling.rename(columns={at: f'freq_{at}' for at in self.freq_columns})
        combined = pd.merge(df,rolling, on=['ID','old_index'])
        for at in self.freq_columns_final:
            combined[at] = combined[at] / combined['ICULOS']
        return combined


    def prepare_data(self,df, rolling=False,freq=True, impute=True):
        """
        main function of this class
        :param df: dataframe to process
        :param rolling: boolean, whether to add window columns
        :param freq: boolean, whether to add frequency columns
        :param impute: boolean, whether to impute the data
        :return: processed dataframe for RNN models
        """
        if rolling:
            df = self.add_rolling_window(df)
        if freq:
            df = self.add_frequency(df)
        df = df[df['time_bm']>=-1*(self.seq_len)]
        df = df[self.columns+self.freq_columns_final+['time_bm','ID','Label']]
        if impute:
            df = self.impute_per_patient(df)
        return df
#%%


class PreProcess():
    """
    This Class is used to prepare the data for the non-time series models
    We add 4 types of columns: window columns and frequency columns,SOFA column and unit3 column, as explained in the
    report. and then aggregate the results to get 1 record (row in the dataframe) for each patient
    """
    def __init__(self,df,window_size = 5, time_bm=-10, imputer_path = None, impute=True):
        """
        :param df: data frame to process
        :param window_size: size of the window to calculate number of tests
        :param time_bm: maximum time before sepsis to filter the data frame
        :param imputer_path: path to trained imputer
        """
        self.frequency_used_attributes = ['BaseExcess', 'FiO2', 'pH', 'PaCO2', 'Glucose', 'Lactate', 'PTT']
        self.freq_columns = ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2',
                 'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride',
                 'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate',
                 'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total',
                 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets']
        self.freq_columns_final = [f'freq_{at}' for at in self.freq_columns]
        self.values_used_attributes = ['Hct', 'Glucose', 'Potassium']
        self.constant_attributes = ['ID', 'max_ICULOS', 'Gender']
        self.other_attributes = ['time_bm', 'HR', 'MAP', 'O2Sat', 'Resp', 'SBP', 'ICULOS']
        self.units_attributes = ['Unit1', 'Unit2']
        self.label_attributes = ['Label', 'SepsisLabel']
        self.window_size = window_size
        self.time_bm = time_bm
        self.test_df = self.preprocess(df)
        self.IDS = self.test_df[['ID']]
        self.cols = list(self.test_df.columns)
        self.cols.remove('ID')
        self.cols.remove('Label')
        # # with open(f'Best_features_RF_run.pickle', 'rb') as handle:
        # #     self.features = pickle.load(handle)
        # self.features = ['max_ICULOS', 'freq_pH', 'freq_TroponinI', 'freq_Bilirubin_direct', 'AST__mean', 'Glucose__mean', 'MAP__min', 'freq_Chloride', 'WBC__median', 'freq_BUN', 'O2Sat__std', 'pH__std', 'WBC__min', 'freq_PaCO2', 'freq_Creatinine', 'PTT__mean', 'Hct__min', 'Glucose__std', 'EtCO2__mean', 'freq_FiO2', 'freq_Glucose', 'freq_Fibrinogen', 'Hgb__mean', 'SaO2__mean', 'freq_SaO2', 'O2Sat__mean', 'Platelets__mean']
        if impute:
            self.knn_imp = joblib.load(imputer_path)
            self.X = pd.DataFrame(self.knn_imp.transform(self.test_df[self.cols]), columns=self.cols)
            self.y = self.test_df['Label'].values

    # create frequency columns for some lab variables
    def add_rolling_window(self, df):
        df = df.sort_values(by=['ID', 'ICULOS'], ascending=[True, True])
        rolling = df[['ID'] + self.frequency_used_attributes].groupby('ID').rolling(window=self.window_size,
                                                                                    closed='both').count()
        rolling = rolling.rename(columns={at: f'{self.window_size}w_sum_{at}' for at in self.frequency_used_attributes})
        rolling = rolling[list(rolling.columns)[1:]].reset_index().set_index('level_1')
        combined = df.join(rolling, how='left', rsuffix='r')
        return combined, rolling

    def calc_frequency(self, df):
        df = df.sort_values(by=['ID', 'ICULOS'], ascending=[True, True])
        rolling = df[['ID', 'ICULOS'] + self.freq_columns].groupby(by=['ID'])[
            self.freq_columns].expanding().count().reset_index().rename(columns={'level_1': 'old_index'})
        df = df.reset_index().rename(columns={'index': 'old_index'})
        rolling = rolling.rename(columns={at: f'freq_{at}' for at in self.freq_columns})
        combined = pd.merge(df, rolling, on=['ID', 'old_index'])
        new_cols = [f'freq_{at}' for at in self.freq_columns]
        for at in new_cols:
            combined[at] = combined[at] / combined['ICULOS']
        freq_df = combined[['ID'] + self.freq_columns_final].groupby(by='ID').last().reset_index()
        return freq_df

    def preprocess(self,df):
        df, df_roll = self.add_rolling_window(df)
        freq_df = self.calc_frequency(df)
        frequency_used_attributes_fixed = [f'{self.window_size}w_sum_{x}' for x in self.frequency_used_attributes]
        df = df[df['time_bm']>=self.time_bm]
        # handle Units123
        df['Unit3'] = ( (1*(df['Unit1']+df['Unit2'])<1) |
                              (df['Unit1'].isna() & df['Unit2'].isna()) )*1
        df['Unit1'][df['Unit1'].isna()] = 0
        df['Unit2'][df['Unit2'].isna()] = 0

        # create SOFA attribute
        df['SOFA'] = df['SBP'] <= 100
        df['SOFA'] += df['Resp'] >= 22

        # aggregations

        data_final = df.groupby(['ID', 'Label','max_ICULOS','Gender']).agg({
                                                            'SOFA': 'max', \
                                                            'Unit1': 'max',\
                                                            'Unit2': 'max',\
                                                            'Unit3': 'max',\
                                                            'HR': ['median', 'max','std'],\
                                                            'MAP': ['median', 'min'],\
                                                            'O2Sat': ['mean','std'],\
                                                            'Resp': ['median', 'max','std'],\
                                                            'SBP': ['median', 'min','std'],\
                                                            'Hct': ['median', 'min'],\
                                                            'Potassium': 'mean',\
                                                            'Glucose': ['mean','std'],\
                                                            'Temp': ['mean', 'min'],\
                                                            'DBP': 'mean',\
                                                            'WBC': ['median', 'min','std'],\
                                                            'EtCO2': 'mean',\
                                                            'BaseExcess': 'mean',\
                                                            'HCO3': 'mean',\
                                                            'FiO2': ['mean','std'],\
                                                            'SaO2': 'mean',\
                                                            'AST': 'mean',\
                                                            'Lactate': 'mean',\
                                                            'Magnesium': 'mean',\
                                                            'Phosphate': 'mean',\
                                                            'TroponinI': 'mean',\
                                                            'Hgb': 'mean',\
                                                            'PTT': 'mean',\
                                                            'Platelets': 'mean',\
                                                            'Age': 'mean',\
                                                            'HospAdmTime': 'mean',\
                                                            'pH': ['std','median'],\
                                                            f'{self.window_size}w_sum_BaseExcess': 'mean',\
                                                            f'{self.window_size}w_sum_FiO2': 'mean',\
                                                            f'{self.window_size}w_sum_pH': 'mean',\
                                                            f'{self.window_size}w_sum_PaCO2': 'mean',\
                                                            f'{self.window_size}w_sum_Glucose': 'mean',\
                                                            f'{self.window_size}w_sum_Lactate': 'mean',\
                                                            f'{self.window_size}w_sum_PTT': 'mean'}).reset_index()
        data_final.columns = ['__'.join(col).strip() for col in data_final.columns.values]
        data_final.rename(columns={"ID__": "ID", "Label__": "Label", "max_ICULOS__":"max_ICULOS", "Gender__":"Gender"}, inplace=True)
        data_final['SOFA__max'] += data_final['SBP__median'] <= 100
        data_final['SOFA__max'] += data_final['Resp__median'] >= 22
        data_final = pd.merge(data_final,freq_df, on='ID')
        return data_final

