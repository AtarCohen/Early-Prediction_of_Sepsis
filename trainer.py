
from LSTM_Model import *
import sys
from torch import optim
import torch.nn as nn
import math
import pandas as pd
from sklearn.metrics import f1_score, accuracy
# import wandb
from datetime import datetime
import tqdm


class Trainer:
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)

    def train(self, data_loader, num_epochs, learning_rate,args,early_stop=8):
        number_of_seqs = len(data_loader.sampler)
        number_of_batches = len(data_loader.batch_sampler)
        train_results_list = []
        print(args.dataset + " " + args.group + " " + args.dataset + " dataset " + "split: " + str(test_split))

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
            correct1 = 0
            total1 = 0

            for batch in data_loader:
                batch_input, batch_target, lengths, mask = batch
                mask = mask.to(self.device)
                optimizer.zero_grad()
                lengths = lengths.to(dtype=torch.int64).to(device='cpu')
                predictions = self.model(batch_input, lengths, mask)
                loss = self.ce(predictions, batch_target)
                loss.backward()
                optimizer.step()
                _, predicted1 = torch.max(predictions[0], 1)
                for i in range(len(lengths)):
                    correct1 += (predicted1[i][:lengths[i]] == batch_target_gestures[i][
                                                               :lengths[i]].squeeze()).float().sum().item()
                    total1 += lengths[i]

                pbar.update(1)
            acc = accuracy(batch_target,predicted)
            F1 =f1_score(batch_target,predicted1)
            # schedular.step(acc)
            pbar.close()
            print(colored(
                dt_string, 'green',
                attrs=['bold']) + f"  [epoch {epoch + 1}: train loss = {epoch_loss / number_of_seqs},   "
                                  f"train acc = {acc}, train F1 = {F1}")
            train_results = {"epoch": epoch, "train loss": epoch_loss / number_of_seqs,
                             "train acc": acc, 'train F1': F1}

            # if args.upload: # **controlled by wandb mode
            # wandb.log(train_results)
            train_results_list.append(train_results)
            if (epoch + 1) % eval_rate == 0:
                print(colored("epoch: " + str(epoch + 1) + " model evaluation", 'red', attrs=['bold']))
                results = {"epoch": epoch}
                # results.update(self.evaluate(eval_dict, test_data_loader, list_of_vids))
                # eval_results_list.append(results)

                # if args.upload is True:  # **controlled by wandb mode
                wandb.log(results)

            # ** new:
            if F1 > best_F1 + 1e-2:
                best_F1 = F1
                steps_no_improve = 0
                torch.save(self.model.state_dict(), os.path.join('/models', "F1_model.h5"))
            else:
                steps_no_improve += 1
                if steps_no_improve >= early_stop:
                    break

        # wandb.log({f'best_{k}': v for k, v in best_results.items()})
        # wandb.finish()
        return train_results_list
