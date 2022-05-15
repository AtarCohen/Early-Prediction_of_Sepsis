import sys
import os
import joblib
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, recall_score,precision_score,roc_auc_score
import tqdm
from torch.utils.data import DataLoader
from LSTM_Model import RNN_Model, Dataset, collate_inputs
import torch
import argparse
import random
from DataPreaparators import DataPreparator, create_patients_df


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def predict_and_eval(model,test_data_loader):
    """
    :param model: model to use for prediction
    :param test_data_loader: dataloader of test set
    :return: prints different success measures and returns dataframe with predictions
    """
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
        print('Accuracy Score: ', accuracy_score(all_labels,all_preds))
        print('Recall Score: ', recall_score(all_labels,all_preds))
        print('Precision Score: ', precision_score(all_labels,all_preds))
        print('roc_auc Score: ', roc_auc_score(all_labels,all_preds))

    results = pd.DataFrame({'ID':ids,'SepsisLabel':all_preds})
    return results



if __name__ == "__main__":

    #columns list for preprocess
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

    # test_files= sys.argv[1] #getting directory path
    # patients = os.listdir(test_files) # Files names list in the given directory
    # test_df = create_patients_df(patients,test_files) # create dataframe

    # p = DataPreparator(columns=COLS, freq_columns=ALL_LAB_ATTR)
    # test_df = p.prepare_data(test_df) #process data for RNN models, including imputation
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




