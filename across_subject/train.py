# %%
from metalearning.dataSets.eeg import BCI2000
from metalearning.models.eeg import EEGNet
from metalearning.maml import MAMLHandler
from torch.utils.data import DataLoader
from dvclive import Live
import numpy as np
import random
import torch
import yaml
import os

live = Live(report="html")
os.system("mkdir -p models")
params = yaml.safe_load(open("params.yaml"))["across_subject"]
live.log_params(params)
seed = params["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# %%
device = torch.device(params["device"])
model = EEGNet(num_class=2, num_channels=64, num_time_points=321, drop_out_prob=params["dropout"])
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
physionet_train = BCI2000(trainSubjects, label_mapping, -1.0, 4.0, 0.0, 2.0, filter_setting, params["k_shot"], params["k_query"], 321)

physionet_valid = BCI2000(validSubjects, label_mapping, -1.0, 4.0, 0.0, 2.0, filter_setting, params["k_shot"], params["k_query"], 321)

# training handler
mamlHandler = MAMLHandler(
    model= model,
    device= device,
    trainDataLoader= DataLoader(physionet_train, params["task_batch_size"], shuffle=True, num_workers=1),
    validationDataLoader= DataLoader(physionet_valid, 1, shuffle=True, num_workers=1),
    updateStepsInInnerLoopTrain= params["adapt_steps"],
    updateStepsInInnerLoopValid= params["adapt_steps"],
    adaptLearningRate=params["adapt_lr"],
    epochs= params["epoch"],
    optimizer= torch.optim.Adam(model.parameters(), lr= params["meta_lr"]),
    modelName="across_subject",
    modelSaveDir='models',
    live=live
)

# train
mamlHandler.train()
live.end()
