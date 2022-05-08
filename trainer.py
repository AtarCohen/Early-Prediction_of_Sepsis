import torch

from LSTM_Model import *
import sys
from torch import optim
import torch.nn as nn
import math
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
# import wandb
from datetime import datetime
import tqdm


class Trainer:
    def __init__(self, model, device="cpu"):
        self.model = model.float()
        self.device = device
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)

    def train(self, train_data_loader,test_data_loader, num_epochs, learning_rate,args,early_stop=8,eval_rate=1):
        number_of_seqs = len(train_data_loader.sampler)
        number_of_batches = len(train_data_loader.batch_sampler)
        train_results_list = []
        eval_results_list = []

        # wandb.init(project=args.project, group=args.group,
        #            name="split: " + str(test_split), entity=args.entity,  # ** we added entity, mode
        #            mode=args.wandb_mode)
        # # delattr(args, 'test_split')
        # wandb.config.update(args, allow_val_change=True)
        self.model.train()
        self.model.to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # ** new -
        # schedular = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, threshold=1e-2,
        #                               threshold_mode='abs', verbose=True)
        best_F1 = 0
        steps_no_improve = 0
        for epoch in range(num_epochs):
            pbar = tqdm.tqdm(total=number_of_batches)
            epoch_loss = 0
            mean_epoch_acc = 0
            mean_epoch_f1= 0

            for batch in train_data_loader:
                batch_input, batch_target, lengths, mask = batch
                mask = mask.to(self.device)
                optimizer.zero_grad()
                lengths = lengths.to(dtype=torch.int64).to(device='cpu')
                predictions = self.model(batch_input, lengths, mask)
                loss = self.ce(predictions, batch_target)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(predictions, 1)
                acc = accuracy_score(batch_target,predicted)
                F1 =f1_score(batch_target,predicted)
            # schedular.step(acc)
            mean_epoch_f1 += F1
            mean_epoch_acc += acc
            pbar.close()
            print(f"  [epoch {epoch + 1}: train loss = {epoch_loss / number_of_seqs},   "
                                  f"train acc = {acc}, train F1 = {F1}")
            train_results = {"epoch": epoch, "train loss": epoch_loss / number_of_seqs,
                             "train acc": acc, 'train F1': F1}

            # if args.upload: # **controlled by wandb mode
            # wandb.log(train_results)
            train_results_list.append(train_results)
            if (epoch + 1) % eval_rate == 0:
                print("epoch: " + str(epoch + 1) + " model evaluation")
                results = {"epoch": epoch}
                results.update(self.eval(test_data_loader))
                eval_results_list.append(results)
                print(f"  [epoch {epoch + 1}: train loss = {epoch_loss / number_of_seqs},   "
                      f"train acc = {acc}, train F1 = {F1}")

                # if args.upload is True:  # **controlled by wandb mode
                # wandb.log(results)

            # ** new:
            if F1 > best_F1 + 1e-2:
                best_F1 = F1
                steps_no_improve = 0
                torch.save(self.model.state_dict(), "F1_model.h5")
            else:
                steps_no_improve += 1
                if steps_no_improve >= early_stop:
                    break

        # wandb.log({f'best_{k}': v for k, v in best_results.items()})
        # wandb.finish()
        return train_results_list,eval_results_list


    def eval(self, test_data_loader):
        results = {}
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            self.model.to(self.device)
            for batch in test_data_loader:
                batch_input, batch_target, lengths, mask = batch
                mask = mask.to(self.device)
                lengths = lengths.to(dtype=torch.int64).to(device='cpu')
                predictions = self.model(batch_input, lengths, mask)
                _, predicted = torch.max(predictions, 1)
                all_preds += predicted
                all_labels += batch_target
            results['acc'] = accuracy_score(all_labels,all_preds)
            results['F1'] = f1_score(all_labels,all_preds)
        self.model.train()
        return results





