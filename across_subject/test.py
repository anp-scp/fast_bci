# %%
from metalearning.dataSets.eeg import BCI2000
from metalearning.models.eeg import EEGNet
from metalearning.maml import MAMLHandler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
import random
import torch
import yaml

params = yaml.safe_load(open("params.yaml"))["across_subject"]

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
physionet_test = BCI2000(testSubjects, label_mapping, -1.0, 4.0, 0.0, 2.0, filter_setting, params["k_shot"], params["k_query"], 321)

# training handler
mamlHandler = MAMLHandler(
    model= model,
    device= device,
    trainDataLoader= None,
    validationDataLoader= None,
    updateStepsInInnerLoopTrain= params["adapt_steps"],
    updateStepsInInnerLoopValid= params["adapt_steps"],
    adaptLearningRate=params["adapt_lr"],
    epochs= None,
    optimizer= None,
    modelName="across_subject",
    modelSaveDir='models',
    live=None
)

accs_finetune = []
accs_test = []
runs = 100
for i in tqdm(range(runs), desc="Test Runs"):
    metrics = mamlHandler.test(DataLoader(physionet_test, 1, shuffle=True, num_workers=1))
    accs_test.append(metrics[0])
    accs_finetune.append(metrics[1])
accs_test = np.vstack(accs_test)
accs_finetune = np.vstack(accs_finetune)
print(f"Finetune Accuracy at every step (0th index for before 1st step):")
fine_tune_mean = accs_finetune.mean(axis=0)
fine_tune_std = accs_finetune.std(axis=0)
fine_tine_metrics = [str(round(a*100,2))+u"\u00B1"+str(round(b*100,2)) for (a,b) in zip(fine_tune_mean, fine_tune_std)]
print(fine_tine_metrics)
print(f"Test Accuracy at every step (0th index for before 1st step):")
test_mean = accs_test.mean(axis=0)
test_std = accs_test.std(axis=0)
test_metrics = [str(round(a*100,2))+u"\u00B1"+str(round(b*100,2)) for (a,b) in zip(test_mean, test_std)]
print(test_metrics)