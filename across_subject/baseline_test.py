#%%
from baseline.trainer import TrainerSingleGPU
from metalearning.dataSets.eeg import BCI2000
from torch.utils.data import DataLoader
from baseline.models.eeg import EEGNet
from tqdm.auto import tqdm
from copy import deepcopy
import numpy as np
import random
import torch
import yaml
import os

os.system("mkdir -p models")
params = yaml.safe_load(open("params.yaml"))["across_subject_baseline"]
seed = params["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
#%%

model_pretrained = EEGNet(num_class=2, num_channels=64, num_time_points=321, drop_out_prob=params["dropout"], norm=params["norm"])
checkpoint = torch.load("models/across_subject_baseline.pt")
model_pretrained.load_state_dict(checkpoint["MODEL_STATE"])
print(model_pretrained)
device = params["device"]
# %%
# datasets
testSubjects = list(range(111))[params["test_subject_start"]:params["test_subject_end"]]
label_mapping = params["label_mapping"]
for key in label_mapping:
    label_mapping[key] = [tuple(x) for x in label_mapping[key]]
filter_setting = {
    'l_freq': params["band_pass_lower"],
    'h_freq': params["band_pass_higher"],
    'fir_design': params["band_pass_filter"],
    'skip_by_annotation': 'edge'
}
physionet_test = BCI2000(testSubjects, label_mapping, -1.0, 4.0, 0.0, 2.0, filter_setting, params["k_finetune"], params["k_test"], 321)
physionet_test = DataLoader(physionet_test)
runs = 100
test_acc_over_all_runs = []
finetune_acc_over_all_runs = []

for _ in tqdm(range(runs), desc="Test Runs"):
    test_acc = np.zeros(shape=(len(physionet_test.dataset), params["steps"]+1), dtype=np.float32)
    finetune_acc = np.zeros(shape=(len(physionet_test.dataset), params["steps"]+1), dtype=np.float32)
    for i, (x_finetune, y_finetune, x_test, y_test) in enumerate(physionet_test):
        x_finetune, y_finetune, x_test, y_test = torch.squeeze(x_finetune, dim=0), torch.squeeze(y_finetune, dim=0),\
            torch.squeeze(x_test,dim=0), torch.squeeze(y_test, dim=0)
        x_finetune, y_finetune, x_test, y_test = x_finetune.to(device), y_finetune.to(device), x_test.to(device), y_test.to(device)
        model = deepcopy(model_pretrained)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr= params["lr"])
        loss_criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            model.eval()
            logits = model(x_finetune)
            y_hat = torch.softmax(logits, dim=1).argmax(dim=1)
            correct = torch.eq(y_finetune, y_hat).sum().item()
            finetune_acc[i][0] = correct / len(y_finetune)

            logits = model(x_test)
            y_hat = torch.softmax(logits, dim=1).argmax(dim=1)
            correct = torch.eq(y_test, y_hat).sum().item()
            test_acc[i][0] = correct / len(y_test)
        
        for j in range(1, params["steps"]+1):
            model.train()
            logits = model(x_finetune)
            optimizer.zero_grad()
            loss = loss_criterion(logits, y_finetune)
            loss.backward()
            optimizer.step()

            model.eval()
            logits = model(x_finetune)
            y_hat = torch.softmax(logits, dim=1).argmax(dim=1)
            correct = torch.eq(y_finetune, y_hat).sum().item()
            finetune_acc[i][j] = correct / len(y_finetune)

            logits = model(x_test)
            y_hat = torch.softmax(logits, dim=1).argmax(dim=1)
            correct = torch.eq(y_test, y_hat).sum().item()
            test_acc[i][j] = correct / len(y_test)
        del model

    test_acc = test_acc.mean(axis=0)
    test_acc_over_all_runs.append(test_acc)
    finetune_acc = finetune_acc.mean(axis=0)
    finetune_acc_over_all_runs.append(finetune_acc)
test_acc_over_all_runs = np.vstack(test_acc_over_all_runs)
finetune_acc_over_all_runs = np.vstack(finetune_acc_over_all_runs)

print(f"Validation acc. of checkpoint on epoch {checkpoint['EPOCHS_RUN']} (0 indexed): {checkpoint['BEST_ACC']}")
print(f"Finetune accuracy at every step (0 means before 1st step):")
fine_tune_mean = finetune_acc_over_all_runs.mean(axis=0)
fine_tune_std = finetune_acc_over_all_runs.std(axis=0)
fine_tine_metrics = [str(round(a*100,2))+u"\u00B1"+str(round(b*100,2)) for (a,b) in zip(fine_tune_mean, fine_tune_std)]
print(fine_tine_metrics)
print(f"Test accuracy at every step (0 means before 1st step):")
test_mean = test_acc_over_all_runs.mean(axis=0)
test_std = test_acc_over_all_runs.std(axis=0)
test_metrics = [str(round(a*100,2))+u"\u00B1"+str(round(b*100,2)) for (a,b) in zip(test_mean, test_std)]
print(test_metrics)
# %%
