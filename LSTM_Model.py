import torch
import torch.nn as nn
import copy
import pandas as pd
from typing import List
import numpy as np
import os
from torch.utils.data import Dataset
import re



class RNN_Model(nn.Module):
    """
    RNN model base class
    Supports GRU or LSTM layers
    """
    def __init__(self, rnn_type, input_dim, hidden_dim=64, bidirectional=False, dropout=0.4, num_layers=2):
        super(RNN_Model, self).__init__()

        self.hidden_dim = hidden_dim
        self.dropout = torch.nn.Dropout(dropout)
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=bidirectional,
                               num_layers=num_layers, dropout=dropout)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=bidirectional,
                              num_layers=num_layers, dropout=dropout)
        # The linear layer that maps from hidden state space to tag space
        self.output = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 2)

    def forward(self, rnn_inputs, lengths, mask):
        rnn_inputs = rnn_inputs.float()

        packed_input = torch.nn.utils.rnn.pack_padded_sequence(rnn_inputs, lengths=lengths, batch_first=True,
                                                               enforce_sorted=False)
        rnn_output, _ = self.rnn(packed_input)
        unpacked_rnn_out, unpacked_rnn_out_lengths = torch.nn.utils.rnn.pad_packed_sequence(rnn_output,
                                                                                            padding_value=-1,
                                                                                            batch_first=True)
        unpacked_rnn_out = self.dropout(unpacked_rnn_out)
        last_out = torch.stack([unpacked_rnn_out[i, unpacked_rnn_out_lengths[i] - 1, :] for i in range(len(lengths))])
        return self.output(last_out)


#
class Dataset(Dataset):
    """
    Dataset for Patients. the patients_df was created in the PreperaData notebook
    this class saves patient_list, since weve used over and under sample methods, the patient list does not include
    all patient from the majority label and includes the minority label with some repetitions.
    """
    def __init__(self, patients_list: list, patients_df: pd.DataFrame, columns):
        self.patients_df = patients_df
        self.patients_list = patients_list
        self.label = False
        self.columns = columns
        if 'Label' in patients_df.columns:
            self.label = True
            # self.columns.remove('Label')

    def __getitem__(self, item):
        patient_df = self.patients_df[self.patients_df['ID'] == self.patients_list[item]]
        patient_label = None
        if self.label:
            patient_label = patient_df['Label'].values[0]
        patient_data = torch.tensor(patient_df[self.columns].values)
        return patient_data, patient_label, self.patients_list[item]

    def __len__(self):
        return len(self.patients_list)


def collate_inputs(batch):
    """
    Collate function to batch several patients which may have different length of stay in the ICU
    :param batch: items from the dataset
    :return: batch which includes: model tensor inputs, labels, lengths, masks and patient ids for predicting
    """
    input_lengths = []
    batch_features = []
    batch_labels = []
    batch_ids = []
    for sample in batch:
        sample_features = sample[0]
        input_lengths.append(sample_features.shape[0])
        batch_features.append(sample_features)
        batch_labels.append(sample[1])
        batch_ids.append(sample[2])
        # pad
    batch = torch.nn.utils.rnn.pad_sequence(batch_features, batch_first=True)
    # compute mask
    input_masks = batch != 0

    return batch.double(), torch.tensor(batch_labels).type(torch.LongTensor), torch.tensor(input_lengths), input_masks, batch_ids
