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
    def __init__(self, rnn_type, input_dim,num_classes_list, hidden_dim=64, bidirectional=False, dropout=0.4,num_layers=2):
        super(RNN_Model, self).__init__()

        self.hidden_dim = hidden_dim
        self.dropout = torch.nn.Dropout(dropout)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=bidirectional,
                                 num_layers=num_layers)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=bidirectional,
                                 num_layers=num_layers)
        # The linear layer that maps from hidden state space to tag space
        self.output = nn.ModuleList([copy.deepcopy(
            nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, num_classes_list[s]) )
                                 for s in range(len(num_classes_list))])


    def forward(self, rnn_inputs, mask):
        outputs=[]
        rnn_inputs = rnn_inputs.permute(0, 2, 1)
        mask=mask.permute(0,2,1)
        rnn_inputs=self.dropout(rnn_inputs)
        # packed_input = pack_padded_sequence(rnn_inpus, lengths=lengths, batch_first=True, enforce_sorted=False)
        rnn_output, _ = self.rnn(rnn_inputs)
        rnn_output = rnn_output*mask
        # unpacked_rnn_out, unpacked_rnn_out_lengths = torch.nn.utils.rnn.pad_packed_sequence(rnn_output, padding_value=-1, batch_first=True)
        # flat_X = torch.cat([unpacked_ltsm_out[i, :lengths[i], :] for i in range(len(lengths))])
        # unpacked_rnn_out = self.dropout(unpacked_rnn_out)
        rnn_output = self.dropout(rnn_output)
        for output_head in self.output_heads:
            outputs.append([output_head(rnn_output).permute(0, 2, 1)])
        return outputs




# dataset for reading feature files we created in feature_extractor.py
class Dataset(Dataset):
    def __init__(self, patients: pd.DataFrame):
        self.patients_df = patients
        self.patients_list = patients.ID.values()
        self.label=False
        self.columns = patients.columns
        if 'Label' in patients.columns:
            self.label=True
            self.columns.remove('Label')




    def __getitem__(self, item):
        patient_df = self.patients_df[self.patients_list['ID']==self.patients_list[item]]
        if self.label:
            patient_label = patient_df['Label']
        patient_data = torch.tensor(patient_df[self.columns])
        return patient_data, patient_label

    def __len__(self):
        return len(self.surgery_folders)


# collate function to pad a batch of surgeries together
def collate_inputs(batch):
    print(batch)
    input_lengths = []
    input_masks = []
    batch_features = {}
    batch_labels = {}
    for sample in batch:
        sample_features = sample[0]
        input_lengths.append(sample_features.shape[0])
        # pad
    batch = torch.nn.utils.rnn.pad_sequence(batch_features[input_name], batch_first=True)
    # compute mask
    input_masks.append(batch_features[input_name] != 0)

    for input_name in label_names:
        batch_labels[input_name] = []
        input_lengths_tmp = []
        for sample in batch:
            sample_labels = sample[1][input_name]
            input_lengths_tmp.append(sample_labels.shape[0])
            batch_labels[input_name].append(sample_labels)
        # pad
        batch_labels[input_name] = torch.nn.utils.rnn.pad_sequence(batch_labels[input_name], padding_value=-100,
                                                                   batch_first=True)
        input_lengths.append(input_lengths_tmp)

    # sanity check
    assert [input_lengths[0]] * len(input_lengths) == input_lengths
    return batch_features, batch_labels, torch.tensor(input_lengths[0]), input_masks[0]
