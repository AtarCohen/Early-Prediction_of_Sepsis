import pandas as pd
import matplotlib.pyplot as plt
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE ##TODO: Insert to ENV.yml
from imblearn.under_sampling import RandomUnderSampler
import random
seed=42
random.seed(seed)
from xgboost import XGBClassifier
#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import pickle
pd.options.mode.chained_assignment = None  # default='warn'

class feature_selection():
    def __init__(self,train_path,validation_path,features=None,method = 'asc', window_size = 5, time_bm=-10,p_smote=0.5, p_under=0.5, model='RF'):
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
        self.train_df = self.preprocess(pd.read_csv(train_path))
        self.columns =  list(self.train_df.columns)
        self.columns.remove('Label')
        self.columns.remove('ID')
        if features == None:
            self.features = self.columns
        self.feature_number = len(self.features)
        self.imputation_with_KNNimputer()
        self.X_train = pd.DataFrame(self.knn_imp.transform(self.train_df[self.features]), columns=self.features)
        self.y_train = self.train_df['Label']
        self.val_df = self.preprocess(pd.read_csv(validation_path))
        self.X_val = pd.DataFrame(self.knn_imp.transform(self.val_df[self.features]), columns=self.features)
        self.y_val = self.val_df['Label']
        # self.test_df = self.preprocess(pd.read_csv(test_path))
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
    def add_rolling_window(self,df):
        df = df.sort_values(by=['ID','ICULOS'], ascending =[True,True])
        rolling = df[['ID']+self.frequency_used_attributes].groupby('ID').rolling(window=self.window_size, closed='both').count()
        rolling= rolling.rename(columns={at: f'{self.window_size}w_sum_{at}' for at in self.frequency_used_attributes})
        rolling=rolling[list(rolling.columns)[1:]].reset_index().set_index('level_1')
        combined = df.join(rolling,how='left', rsuffix= 'r')
        return combined, rolling


    def preprocess(self,df):
        df, df_roll = self.add_rolling_window(df)
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
        return data_final

    def imputation_with_KNNimputer(self, n=3):
        data_knn_imputed = self.train_df[self.columns].copy(deep=True)    # Copy the data
        self.knn_imp = KNNImputer(n_neighbors=n) # Init the transformer
        self.knn_imp.fit(data_knn_imputed)

    def os_with_smote(self,X,y):
        os = SMOTE(sampling_strategy=self.p_smote, random_state=0)
        columns = X.columns
        os_data_X, os_data_y = os.fit_resample(X, y)
        os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
        os_data_y = pd.DataFrame(data=os_data_y,columns=['Label'])
        return os_data_X, os_data_y

    def feature_selection_asc(self):
        best_features = []
        best_f1 = -1
        for i in range(self.feature_number):
            res = {'Train':{},'Val':{}}
            print('*'*10,f'Adding Feature {i+1}','*'*10)
            best_i_feature = ''
            best_i_f1 = -1
            for f in self.features:
                cols = [f]+self.chosen_features
                X = self.X_train[cols]
                X, y = self.os_with_smote(X,self.y_train)
                under = RandomUnderSampler(sampling_strategy=self.p_under)
                X, y = under.fit_resample(X, y)
                #Train model
                self.set_seed()
                self.model.fit(X,y)
                y_train_pred = self.model.predict(X)
                res['Train'][f] = f1_score(y,y_train_pred)
                X_val = self.X_val[cols]
                y_val_pred = self.model.predict(X_val)
                val_f1 = f1_score(self.y_val,y_val_pred)
                res['Val'][f] = val_f1
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
            self.res['Train'][i]=res['Train']
            self.res['Val'][i]=res['Val']
        self.best_features = best_features
        self.best_f1 = best_f1
        print(f'Best F1 Score: {best_f1}')
        print(f'Best Features: {best_features}')
        with open('Results.pickle', 'wb') as handle:
            pickle.dump(self.res, handle, protocol=pickle.HIGHEST_PROTOCOL)


train_path="/home/student/filtered_train_df_0705.csv"
validation_path = "/home/student/filtered_val_df_0705.csv"
F_selector = feature_selection(train_path = train_path, validation_path=validation_path)