import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import matplotlib.pyplot as plt

plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE  ##TODO: Insert to ENV.yml
from imblearn.under_sampling import RandomUnderSampler
import random
import argparse

seed = 42
random.seed(seed)
from xgboost import XGBClassifier
# Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import pickle
import wandb
import joblib

pd.options.mode.chained_assignment = None  # default='warn'
pd.reset_option('all')


class NotSerialModelsTrainer():
    def __init__(self,args):
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
        self.window_size = args.window_size
        self.time_bm = args.time_bm
        self.p_smote = args.over_sample_rate
        self.p_under = args.under_sample_rate
        self.train_df = self.preprocess(pd.read_csv(args.train_path))
        self.columns =  list(self.train_df.columns)
        self.columns.remove('Label')
        self.columns.remove('ID')
        self.method = args.selector_method
        if args.impute:
            print('-'*10,'Training Imputer','-'*10)
            self.imputation_with_KNNimputer(path= args.impute_path)
        else:
            print('-'*10,'Loading Imputer','-'*10)
            self.knn_imp = joblib.load(args.impute_path)
        self.X_train = pd.DataFrame(self.knn_imp.transform(self.train_df[self.columns]), columns=self.columns)
        self.y_train = self.train_df['Label'].values
        self.val_df = self.preprocess(pd.read_csv(args.validation_path))
        self.X_val = pd.DataFrame(self.knn_imp.transform(self.val_df[self.columns]), columns=self.columns)
        self.y_val = self.val_df['Label'].values
        self.test_df =self.preprocess(pd.read_csv(args.test_path))
        self.X_test = pd.DataFrame(self.knn_imp.transform(self.test_df[self.columns]), columns=self.columns)
        self.y_test=self.test_df['Label'].values
        self.res = {'Train': {}, 'Val': {}}
        self.model_name = args.model
        self.model_path = f'{self.model_name}_{args.run_id}'


    def set_model(self):
        if self.model_name == 'LR':
            self.model = LogisticRegression()
        elif self.model_name == 'RF':
            self.model = RandomForestClassifier(n_estimators=100, random_state=0)
        elif self.model_name == 'XGB':
            self.model = XGBClassifier()

    def set_seed(self, seed=42):
        random.seed(seed)

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

    def preprocess(self, df):
        df, df_roll = self.add_rolling_window(df)
        freq_df = self.calc_frequency(df)
        frequency_used_attributes_fixed = [f'{self.window_size}w_sum_{x}' for x in self.frequency_used_attributes]
        df = df[df['time_bm'] >= self.time_bm]
        # handle Units123
        df['Unit3'] = ((1 * (df['Unit1'] + df['Unit2']) < 1) |
                       (df['Unit1'].isna() & df['Unit2'].isna())) * 1
        df['Unit1'][df['Unit1'].isna()] = 0
        df['Unit2'][df['Unit2'].isna()] = 0

        # create SOFA attribute
        df['SOFA'] = df['SBP'] <= 100
        df['SOFA'] += df['Resp'] >= 22

        # aggregations

        data_final = df.groupby(['ID', 'Label', 'max_ICULOS', 'Gender']).agg({
            'SOFA': 'max', \
            'Unit1': 'max', \
            'Unit2': 'max', \
            'Unit3': 'max', \
            'HR': ['median', 'max', 'std'], \
            'MAP': ['median', 'min'], \
            'O2Sat': ['mean', 'std'], \
            'Resp': ['median', 'max', 'std'], \
            'SBP': ['median', 'min', 'std'], \
            'Hct': ['median', 'min'], \
            'Potassium': 'mean', \
            'Glucose': ['mean', 'std'], \
            'Temp': ['mean', 'min'], \
            'DBP': 'mean', \
            'WBC': ['median', 'min', 'std'], \
            'EtCO2': 'mean', \
            'BaseExcess': 'mean', \
            'HCO3': 'mean', \
            'FiO2': ['mean', 'std'], \
            'SaO2': 'mean', \
            'AST': 'mean', \
            'Lactate': 'mean', \
            'Magnesium': 'mean', \
            'Phosphate': 'mean', \
            'TroponinI': 'mean', \
            'Hgb': 'mean', \
            'PTT': 'mean', \
            'Platelets': 'mean', \
            'Age': 'mean', \
            'HospAdmTime': 'mean', \
            'pH': ['std', 'median'], \
            f'{self.window_size}w_sum_BaseExcess': 'mean', \
            f'{self.window_size}w_sum_FiO2': 'mean', \
            f'{self.window_size}w_sum_pH': 'mean', \
            f'{self.window_size}w_sum_PaCO2': 'mean', \
            f'{self.window_size}w_sum_Glucose': 'mean', \
            f'{self.window_size}w_sum_Lactate': 'mean', \
            f'{self.window_size}w_sum_PTT': 'mean'}).reset_index()
        data_final.columns = ['__'.join(col).strip() for col in data_final.columns.values]
        data_final.rename(
            columns={"ID__": "ID", "Label__": "Label", "max_ICULOS__": "max_ICULOS", "Gender__": "Gender"},
            inplace=True)
        data_final['SOFA__max'] += data_final['SBP__median'] <= 100
        data_final['SOFA__max'] += data_final['Resp__median'] >= 22
        data_final = pd.merge(data_final, freq_df, on='ID')
        return data_final

    def imputation_with_KNNimputer(self, n=3, path=None):
        data_knn_imputed = self.train_df[self.columns].copy(deep=True)  # Copy the data
        self.knn_imp = KNNImputer(n_neighbors=n)  # Init the transformer
        self.knn_imp.fit(data_knn_imputed)
        if path:
            joblib.dump(self.knn_imp,path)

    def os_with_smote(self, X, y):
        os = SMOTE(sampling_strategy=self.p_smote, random_state=0)
        columns = X.columns
        os_data_X, os_data_y = os.fit_resample(X, y)
        os_data_X = pd.DataFrame(data=os_data_X, columns=columns)
        os_data_y = pd.DataFrame(data=os_data_y, columns=['Label'])
        return os_data_X, os_data_y


    def activate_selector(self, method='asc'):
        self.method = method
        self.restart_selector()
        self.feature_number = len(self.columns)
        if self.method=='asc':
            self.feature_selection_asc()


    def restart_selector(self):
        self.columns = list(self.train_df.columns)
        self.columns.remove('Label')
        self.columns.remove('ID')
        self.features = self.columns
        if self.method == 'asc':
            self.chosen_features = []
        else:
            self.chosen_features = self.features.copy()


    def train_model(self,cols=None, save=False):
        if not cols:
            cols=self.features
        X = self.X_train[cols]
        X, y = self.os_with_smote(X, self.y_train)
        under = RandomUnderSampler(sampling_strategy=self.p_under)
        X, y = under.fit_resample(X, y)
        # Train model
        self.set_model()
        self.set_seed()
        self.model.fit(X, y.values.ravel())
        if save:
            joblib.dump(self.model, self.model_path)
        y_train_pred= self.model.predict(X)
        return f1_score(y, y_train_pred)

    def eval(self, cols, ds='val'):
        X = self.X_val[cols] if ds=='val' else self.X_test[cols]
        # X_val = self.X_val[cols]
        y_pred = self.model.predict(X)
        y = self.y_val if ds=='val' else self.y_test
        val = f1_score(y, y_pred)
        return val

    def feature_selection_asc(self):
        best_features = []
        best_f1 = -1
        f1_test_on_best=-1
        best_features_dict = {}
        for i in range(self.feature_number):
            res = {'Train': {}, 'Val': {}}
            print('*' * 10, f'Adding Feature {i + 1}', '*' * 10)
            best_i_feature = ''
            best_i_f1 = -1
            for f in self.features:
                cols = self.chosen_features+[f]
                res['Train'][f] = self.train_model(cols)
                val_f1 = self.eval(cols, ds='val')
                res['Val'][f] = val_f1
                if val_f1 > best_i_f1:
                    best_i_f1 = val_f1
                    best_i_feature = f
                    f1_test = self.eval(cols, ds='test')
            print(f'Best F1 Score for round {i + 1}: {best_i_f1}')
            print(f'Feature Added: {best_i_feature}')
            self.chosen_features += [best_i_feature]
            best_features_dict[i+1] = self.chosen_features
            self.features.remove(best_i_feature)
            wandb.log({'Epoch': i + 1, 'F1 val': best_i_f1, 'Feature': best_i_feature, 'F1 Test':f1_test})
            if best_i_f1 > best_f1:
                best_f1 = best_i_f1
                best_features = self.chosen_features
                f1_test_on_best = f1_test
                joblib.dump(self.model, f'best_{self.model_name}_{i+1}')
            self.res['Train'][i] = res['Train']
            self.res['Val'][i] = res['Val']
        self.best_features = best_features
        self.best_f1 = best_f1
        print(f'Best F1 Score: {best_f1}')
        print(f'Best Features: {best_features}')
        print(f'F1 Test on Best Features: {f1_test_on_best}')

        with open(f'Results_{self.model_name}_run.pickle', 'wb') as handle:
            pickle.dump(self.res, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'Best_features_{self.model_name}_run.pickle', 'wb') as handle:
            pickle.dump(self.best_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'Best_features_dict_{self.model_name}_run.pickle', 'wb') as handle:
            pickle.dump(best_features_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        wandb.log({'best_features': self.best_features})




def parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id',  default='1205', type=str)
    parser.add_argument('--wandb_mode', choices=['online', 'offline', 'disabled'], default='online', type=str)
    parser.add_argument('--project', default="Sepsis_Predictions", type=str)

    parser.add_argument('--model', choices=['RF','XGB','LR'], default='RF', type=str)
    parser.add_argument('--mode', choices=['selector','trainer'], default='selector', type=str)
    parser.add_argument('--selector_method', choices=['asc','dsc'], default='asc', type=str)
    parser.add_argument('--impute_path', default='knn_imputer', type=str)
    parser.add_argument('--impute', default=False, type=bool)
    parser.add_argument('--train_path', default="/home/student/filtered_train_df_0705.csv", type=str)
    parser.add_argument('--validation_path', default= "/home/student/filtered_val_df_0705.csv", type=str)
    parser.add_argument('--test_path', default= "/home/student/filtered_test_df_0705.csv", type=str)

    parser.add_argument('--over_sample_rate', default='0.5', type=float)
    parser.add_argument('--under_sample_rate', default='0.5', type=float)
    parser.add_argument('--window_size', default='5', type=int)
    parser.add_argument('--time_bm', default='-10', type=int)


    args = parser.parse_args()
    return args



args = parsing()
trainer = NotSerialModelsTrainer(args)
if args.mode=='trainer':
    with open(f'Best_features_{args.model}_run.pickle', 'rb') as handle:
        features = pickle.load(handle)
    train_f1 = trainer.train_model(save=True, cols=features)
    print(f'Train F1 Score: {train_f1}')
    print(f'Val F1 Score: {trainer.eval(cols=features, ds="val")}')
if args.mode=='selector':
    wandb.init(project="Non_Serial_Feature_Selection_run2", entity="labteam", mode='online',
               name=f'{args.model}_Asc Features')
    trainer.activate_selector()