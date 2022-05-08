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
    def __init__(self, rnn_type, input_dim, hidden_dim=64, bidirectional=False, dropout=0.4,num_layers=2):
        super(RNN_Model, self).__init__()

        self.hidden_dim = hidden_dim
        # self.dropout = torch.nn.Dropout(dropout)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=bidirectional,
                                 num_layers=num_layers, dropout=dropout)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=bidirectional,
                                 num_layers=num_layers, dropout = dropout)
        # The linear layer that maps from hidden state space to tag space
        self.output = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 2)

    def forward(self, rnn_inputs,lengths, mask):
        outputs=[]
        # rnn_inputs = rnn_inputs.permute(0, 2, 1)
        # mask=mask.permute(0,2,1)
        # rnn_inputs=self.dropout(rnn_inputs)
        rnn_inputs = rnn_inputs.float()
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(rnn_inputs, lengths=lengths, batch_first=True, enforce_sorted=False)
        rnn_output, _ = self.rnn(packed_input)
        # rnn_output = rnn_output*rnn_output*mask[:,:,0].unsqueeze(2)
        unpacked_rnn_out, unpacked_rnn_out_lengths = torch.nn.utils.rnn.pad_packed_sequence(rnn_output, padding_value=-1, batch_first=True)
        last_out = torch.stack([unpacked_rnn_out[i, unpacked_rnn_out_lengths[i] - 1, :] for i in range(len(lengths))])
        # flat_X = torch.cat([unpacked_ltsm_out[i, :lengths[i], :] for i in range(len(lengths))])
        # unpacked_rnn_out = self.dropout(unpacked_rnn_out)
        # rnn_output = self.dropout(rnn_output)
        return  self.output(last_out)





# dataset for reading feature files we created in feature_extractor.py
class Dataset(Dataset):
    def __init__(self, patients_list:list, patients_df: pd.DataFrame):
        self.patients_df = patients_df
        self.patients_list = patients_list
        self.label=False
        self.columns = ['HR','Max_ICULOS']
        if 'Label' in patients.columns:
            self.label=True
            # self.columns.remove('Label')




    def __getitem__(self, item):
        patient_df = self.patients_df[self.patients_df['ID']==self.patients_list[item]]
        if self.label:
            patient_label = patient_df['Label'].values[0]
        patient_data = torch.tensor(patient_df[self.columns].values)
        return patient_data, patient_label

    def __len__(self):
        return len(self.patients_list)


# collate function to pad a batch of surgeries together
def collate_inputs(batch):
    print(batch)
    input_lengths = []
    input_masks = []
    batch_features = []
    batch_labels = []
    for sample in batch:
        sample_features = sample[0]
        input_lengths.append(sample_features.shape[0])
        batch_features.append(sample_features)
        batch_labels.append(sample[1])
        # pad
    batch = torch.nn.utils.rnn.pad_sequence(batch_features, batch_first=True)
    # compute mask
    input_masks= batch != 0

    # for input_name in label_names:
    #     batch_labels[input_name] = []
    #     input_lengths_tmp = []
    #     for sample in batch:
    #         sample_labels = sample[1][input_name]
    #         input_lengths_tmp.append(sample_labels.shape[0])
    #         batch_labels[input_name].append(sample_labels)
    #     # pad
    #     batch_labels[input_name] = torch.nn.utils.rnn.pad_sequence(batch_labels[input_name], padding_value=-100,
    #                                                                batch_first=True)
    #     input_lengths.append(input_lengths_tmp)

    # sanity check
    return batch.double(), torch.tensor(batch_labels), torch.tensor(input_lengths), input_masks
