import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE ##TODO: Insert to ENV.yml
from imblearn.under_sampling import RandomUnderSampler
import statsmodels.api as sm
import random
seed=42
random.seed(seed)
from xgboost import XGBClassifier
#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


class feature_selection():
    def __init__(self,train_path,validation_path, features,method = 'asc', window_size = 5, time_bm=-10,p_smote=0.5, p_under=0.5, model='randomforest'):
        self.train_df = self.preprocess(pd.read_csv(train_path))
        self.val_df = self.preprocess(pd.read_csv(validation_path))
        # self.test_df = self.preprocess(pd.read_csv(test_path))
        self.features = features #features to select from
        self.frequency_used_attributes = ['BaseExcess', 'FiO2', 'pH', 'PaCO2', 'Glucose', 'Lactate', 'PTT']
        self.values_used_attributes = ['Hct', 'Glucose', 'Potassium']
        self.constant_attributes = ['ID', 'max_ICULOS', 'Gender']
        self.other_attributes = ['time_bm', 'HR', 'MAP', 'O2Sat', 'Resp', 'SBP', 'ICULOS']
        self.units_attributes = ['Unit1', 'Unit2']
        self.label_attributes = ['Label', 'SepsisLabel']
        self.window_size = window_size
        self.time_bm = time_bm
        self.p_smote = p_smote
        self.p_under = p_under
        if model=='logreg':
            self.model = LogisticRegression()
        elif model=='RF':
            self.model = RandomForestClassifier(n_estimators=100, random_state=0)
        elif model=='XGB':
            self.model = XGBClassifier()
        self.res = {'Train': {}, 'Val': {}}
        if method=='asc':
            self.chosen_features = []
            self.feature_selection_asc()
        else:
            self.chosen_features = features

    def set_seed(self,seed=42):
        random.seed(seed)

    # create frequency columns for some lab variables
    def add_rolling_window(self,df, attr, window_size):
        df = df.sort_values(by=['ID','ICULOS'], ascending =[True,True])
        rolling = df[['ID']+attr].groupby('ID').rolling(window=window_size, closed='both').count()
        rolling= rolling.rename(columns={at: f'{window_size}w_sum_{at}' for at in attr})
        rolling=rolling[list(rolling.columns)[1:]].reset_index().set_index('level_1')
        combined = df.join(rolling,how='left', rsuffix= 'r')
        return combined, rolling

    def preprocess(self,df):
        df_with_roll, df_roll = self.add_rolling_window(df,self.frequency_used_attributes,self.window_size)
        frequency_used_attributes_fixed = [f'{self.window_size}w_sum_{x}' for x in self.frequency_used_attributes]
        df_with_roll = df_with_roll[df_with_roll['time_bm']>=self.time_bm]
        # handle Units123
        df_with_roll['Unit3'] = ( (1*(df_with_roll['Unit1']+df_with_roll['Unit2'])<1) |
                              (df_with_roll['Unit1'].isna() & df_with_roll['Unit2'].isna()) )*1
        df_with_roll['Unit1'][df_with_roll['Unit1'].isna()] = 0
        df_with_roll['Unit2'][df_with_roll['Unit2'].isna()] = 0
        df_with_roll[['Unit1','Unit2','Unit3']]
    # aggregations
        data_final = df_with_roll.groupby(['ID', 'Label','max_ICULOS','Gender']).agg({
                                                            'Unit1': 'max',\
                                                            'Unit2': 'max',\
                                                            'Unit3': 'max',\
                                                            'HR': ['median', 'max'],\
                                                            'MAP': ['median', 'min'],\
                                                            'O2Sat': ['mean'],\
                                                            'Resp': ['median', 'max'],\
                                                            'SBP': ['median', 'min'],\
                                                            'Hct': ['median', 'min'],\
                                                            'Potassium': 'mean',\
                                                            'Glucose': 'mean',\
                                                            'Temp': ['mean', 'min'],\
                                                            'DBP': 'mean',\
                                                            'WBC': ['median', 'min'],\
                                                            'EtCO2': 'mean',\
                                                            'BaseExcess': 'mean',\
                                                            'HCO3': 'mean',\
                                                            'FiO2': 'mean',\
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
                                                            f'{self.window_size}w_sum_BaseExcess': 'mean',\
                                                            f'{self.window_size}w_sum_FiO2': 'mean',\
                                                            f'{self.window_size}w_sum_pH': 'mean',\
                                                            f'{self.window_size}w_sum_PaCO2': 'mean',\
                                                            f'{self.window_size}w_sum_Glucose': 'mean',\
                                                            f'{self.window_size}w_sum_Lactate': 'mean',\
                                                            f'{self.window_size}w_sum_PTT': 'mean'}).reset_index()
        data_final.columns = ['__'.join(col).strip() for col in data_final.columns.values]
        data_final.rename(columns={"ID__": "ID", "Label__": "Label", "max_ICULOS__":"max_ICULOS", "Gender__":"Gender"}, inplace=True)
        return data_final

    def imputation_with_KNNimputer(self,df, n=3):
        data_knn_imputed = df.copy(deep=True)    # Copy the data
        self.knn_imp = KNNImputer(n_neighbors=n) # Init the transformer
        self.knn_imp.fit(data_knn_imputed)

    def os_with_smote(self,df):
        X = df.loc[:, df.columns != 'Label']
        y = df.loc[:, df.columns == 'Label']
        os = SMOTE(sampling_strategy=self.p_smote, random_state=0)
        columns = X.columns
        os_data_X, os_data_y = os.fit_resample(X, y)
        os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
        os_data_y = pd.DataFrame(data=os_data_y,columns=['Label'])
        return os_data_X, os_data_y

    def feature_selection_asc(self):
        best_features = []
        best_f1 = -1
        for i in range(len(self.features)):
            print('*'*10,f'Adding Feature {i+1}','*'*10)
            best_i_feature = ''
            best_i_f1 = -1
            for f in self.features:
                cols = [f]+self.chosen_features
                X = self.train_df[cols]
                y = self.train_df['Label']
                X = pd.DataFrame(self.knn_imp.transform(X), columns=cols)
                X, y = self.os_with_smote(X,y)
                under = RandomUnderSampler(sampling_strategy=self.p_under)
                X, y = under.fit_resample(X, y)
                #Train model
                self.set_seed()
                self.model.fit(X,y)
                y_train_pred = self.model.predict(X)
                self.res['Train'][f] = f1_score(y,y_train_pred)
                X_val = self.val_df[cols]
                y_val = self.val_df['Label']
                X_val =  pd.DataFrame(self.knn_imp.transform(X_val), columns=cols)
                y_val_pred = self.model.predict(X_val)
                val_f1 = f1_score(y_val,y_val_pred)
                self.res['Val'][f] = val_f1
                if val_f1>best_i_f1:
                    best_i_f1=val_f1
                    best_i_feature = f
            print(f'Best F1 Score for round {i+1}: {best_i_f1}')
            print(f'Feature Added: {best_i_feature}')
            self.chosen_features += [best_i_feature]
            self.features.remove(f)
            if best_i_f1 > best_f1:
                best_f1 = best_i_f1
                best_features = self.chosen_features
        self.best_features = best_features
        self.best_f1 = best_f1
        print(f'Best F1 Score: {best_f1}')
        print(f'Best Features: {best_features}')



