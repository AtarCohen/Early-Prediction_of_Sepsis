import torch

from LSTM_Model import *
import sys
from torch import optim
import torch.nn as nn
import math
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
import wandb
from datetime import datetime
import tqdm


class Trainer:
    """
    RNN Trainer class
    """
    def __init__(self, model, device="cpu"):
        """
        :param model: RNN model
        :param device: in our case, cpu
        """
        self.model = model.float()
        self.device = device
        self.ce = nn.CrossEntropyLoss(ignore_index=-100) #loss

    def train(self, train_data_loader,test_data_loader, num_epochs, learning_rate,args,early_stop=15,eval_rate=1):
        """
        :param train_data_loader:
        :param test_data_loader:
        :param num_epochs: max number of epochs
        :param learning_rate: lr for the optimizer
        :param args:
        :param early_stop: number of epochs with no F1 on train improvement to stop running
        :param eval_rate: validation evaluation rate
        :return: results for each epoch and each dataset
        """
        number_of_seqs = len(train_data_loader.sampler)
        number_of_batches = len(train_data_loader.batch_sampler)
        train_results_list = []
        eval_results_list = []

        wandb.init(project="lab1_lstm", entity="labteam",mode=args.wandb_mode) #logging to wandb
        wandb.config.update(args)
        self.model.train()
        best_results = {'val acc': 0, 'val F1': 0}
        self.model.to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        best_F1 = 0
        steps_no_improve = 0
        #Train model
        for epoch in range(num_epochs):
            pbar = tqdm.tqdm(total=number_of_batches)
            epoch_loss = 0
            mean_epoch_acc = 0
            all_targets = torch.Tensor()
            all_predictions = torch.Tensor()
            for batch in train_data_loader:
                batch_input, batch_target, lengths, mask, _ = batch
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
                mean_epoch_acc +=acc
                all_targets = torch.cat([all_targets,batch_target])
                all_predictions = torch.cat([all_predictions,predicted])

            all_targets = all_targets.detach().numpy()
            all_predictions = all_predictions.detach().numpy()
            epoch_F1= f1_score(all_targets,all_predictions)
            epoch_acc = accuracy_score(all_targets,all_predictions)
            pbar.close()
            print(f" \n [epoch {epoch + 1}: train loss = {epoch_loss / number_of_seqs},  mean acc: {mean_epoch_acc/number_of_batches} "
                                  f"train acc = {epoch_acc}, train F1 = {epoch_F1}")
            train_results = {"epoch": epoch, "train loss": epoch_loss / number_of_seqs,
                             "train acc": epoch_acc, 'train F1': epoch_F1}
            wandb.log(train_results)
            train_results_list.append(train_results)
            if (epoch + 1) % eval_rate == 0:
                print("epoch: " + str(epoch + 1) + " model evaluation")
                results = {"epoch": epoch}
                results.update(self.eval(test_data_loader))
                eval_results_list.append(results)
                print(f"\n  [epoch {epoch + 1}: val acc = {results['val acc']}, val F1 = {results['val F1']}")
                wandb.log(results)

                if results['val acc'] > best_results['val acc'] + 5e-3:
                    best_results['val acc'] = results['val acc']
                    best_results['epoch acc'] = epoch

                if results['val F1'] > best_results['val F1'] + 5e-3:
                    best_results['val F1'] = results['val F1']
                    best_results['epoch F1'] = epoch
                    torch.save({"model_state": self.model.state_dict()}, f'{wandb.run.dir}/best_f1.pth') #save model with best Val F1

            if epoch_F1 > best_F1 + 1e-2:
                best_F1 = epoch_F1
                steps_no_improve = 0
            else:
                steps_no_improve += 1
                if steps_no_improve >= early_stop:
                    break

        wandb.log({f'best_{k}': v for k, v in best_results.items()}) #log best results
        wandb.finish()
        return train_results_list,eval_results_list


    def eval(self, test_data_loader,name='val'):
        results = {}
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            self.model.to(self.device)
            for batch in test_data_loader:
                batch_input, batch_target, lengths, mask,_ = batch
                mask = mask.to(self.device)
                lengths = lengths.to(dtype=torch.int64).to(device='cpu')
                predictions = self.model(batch_input, lengths, mask)
                _, predicted = torch.max(predictions, 1)
                all_preds += predicted
                all_labels += batch_target
            results[f'{name} acc'] = accuracy_score(all_labels,all_preds)
            results[f'{name} F1'] = f1_score(all_labels,all_preds)
        self.model.train()
        return results






