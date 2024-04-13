#%%
from baseline.trainer import TrainerSingleGPU
from baseline.datasets.eeg import BCI2000
from torch.utils.data import DataLoader
from baseline.models.eeg import EEGNet
from dvclive.live import Live
import numpy as np
import random
import torch
import yaml
import os

live = Live(report="html")
os.system("mkdir -p models")
params = yaml.safe_load(open("params.yaml"))["across_subject_baseline"]
live.log_params(params)
seed = params["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
#%%

model = EEGNet(num_class=2, num_channels=64, num_time_points=321, drop_out_prob=params["dropout"], norm=params["norm"])
print(model)

# %%
# datasets
trainSubjects = list(range(111))[params["train_subject_start"]:params["train_subject_end"]]
validSubjects = list(range(111))[params["valid_subject_start"]:params["valid_subject_end"]]
label_mapping = params["label_mapping"]
for key in label_mapping:
    label_mapping[key] = [tuple(x) for x in label_mapping[key]]
filter_setting = {
    'l_freq': params["band_pass_lower"],
    'h_freq': params["band_pass_higher"],
    'fir_design': params["band_pass_filter"],
    'skip_by_annotation': 'edge'
}

if params["style"] == "cross":
    physionet_train = BCI2000(trainSubjects, label_mapping, -1.0, 4.0, 0.0, 2.0, filter_setting, params["num_samples"], 321)

    physionet_valid = BCI2000(validSubjects, label_mapping, -1.0, 4.0, 0.0, 2.0, filter_setting, params["num_samples"], 321)
else:
    physionet = BCI2000(trainSubjects, label_mapping, -1.0, 4.0, 0.0, 2.0, filter_setting, params["num_samples"], 321)
    physionet_train, physionet_valid = torch.utils.data.random_split(physionet, [0.8,0.2])

# training
trainer = TrainerSingleGPU(
    model=model,
    train_data= DataLoader(physionet_train, params["batch_size"], shuffle=True, num_workers=1),
    valid_data= DataLoader(physionet_valid, 1, shuffle=True, num_workers=1),
    optimizer= torch.optim.Adam(model.parameters(), lr= params["lr"]),
    gpu_id=params["device"],
    loss_criterion=torch.nn.CrossEntropyLoss(),
    snap_shot_path="models/across_subject_baseline.pt",
    live=live
)

best_valid_acc = trainer.train(params["epoch"])
live.log_metric("best_validation_acc", best_valid_acc, plot=False)
live.end()
# %%
