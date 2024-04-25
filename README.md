# Fast BCI

Pytorch code for the paper [Evaluating Fast Adaptability of Neural Networksfor Brain-Computer Interface](https://arxiv.org/abs/2404.15350) [Accepted in IJCNN 24].

Open `docs/index.html` in a browser for web based documentation.

# Abstract

<div style="text-align: justify">
Electroencephalography (EEG)  classification is a versatile and portable technique for building non-invasive Brain-computer Interfaces (BCI). However, the classifiers that decode cognitive states from EEG brain data perform poorly when tested on newer domains, such as tasks or individuals absent during model training. Researchers have recently used complex strategies like Model-agnostic meta-learning (MAML) for domain adaptation. Nevertheless, there is a need for an evaluation strategy to evaluate the fast adaptability of the models, as this characteristic is essential for real-life BCI applications for quick calibration. We used motor movement and imaginary signals as input to Convolutional Neural Networks (CNN) based classifier for the experiments. Datasets with EEG signals typically have fewer examples and higher time resolution. Even though batch-normalization is preferred for Convolutional Neural Networks (CNN), we empirically show that layer-normalization can improve the adaptability of CNN-based EEG classifiers with not more than ten fine-tuning steps.
In summary, the present work (i) proposes a simple strategy to evaluate fast adaptability, and (ii) empirically demonstrate fast adaptability across individuals as well as across tasks with simple transfer learning as compared to MAML approach.
</div>

# Setup

## Pre-requisite

1. Install conda
2. Install dependencies:

```shell
conda env create -f environment.yml
```

3. Activate the conda environment

```shell
conda activate fast_bci
```

4. Go to root directory of the code
5. Install the package in `development mode`

```shell
conda develop .
```

6. Download the data by running the following command. Running this for the first time may ask for a path for `MNE_DATA`. Set the desired path and continue.

```shell
python3 download_data.py
```

## Directory structure

> Terminology Alert
> 
> 1. Subject: In BCI wrold, a person for whom EEG is recorded is refered as a `subject`.
> 2. In BCI world, a task can be an activity or human body movement. For example: moving hands
> 3. baseline: In the code, `baseline` refers to tranfer learning. In the beginning of the
> experiment, we thought that MAML will work better and named transfer learning to baseline.
> But, later we found that transfer learning works better.

| Directory      | Description                                      |
|----------------|--------------------------------------------------|
| metalearning   | Module for MAML related APIs                     |
| baseline       | Module for transfer learning related APIs        |
| across_subject | Experiments for across individual adaptability   |
| across_task    | Experiments for across activity adaptability     |

> NOTE:
>
> * Web based API docs for `metalearning` module is available in `docs/metalearning/index.html`.
> * Web based API docs for `baseline` module is available in `docs/baseline/index.html`.

## Scripts

1. Training CNN model using MAML:                                           `across_subject/train.py`
2. Testing CNN model trained using MAML on new individuals:                 `across_subject/test.py`
3. Training CNN model using transfer learning:                              `across_subject/baseline_train.py`
4. Testing CNN model trained using transfer learning on new individuals:    `across_subject/baseline_test.py`
5. Testing CNN model trained using MAML on new activities:                  `across_task/test.py`
6. Testing CNN model trained using transfer learning on new activities:     `across_task/baseline_test.py`

The parameters of the scripts are defined in `params.yaml` in the respective directories.

**Label mapping**

We map EEG data with labels using a parameter called `label_mapping`. The annotation of EEG data
is provided in the homepage of [Physionet's EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/).

Here, we describe how to map labels with data in the dataset.

For example, we need to perform binary classification for open and close left vs right fist (Task 1).
Then, as per the annotation, we would need data with code `T1` at runs 3,7,11 for left fist and
data with code `T2` at runs 3,7,11 for right fist. The following `YAML` snippet labels data for left
fist as label `0` and data for right fist as label `1`:

```yaml
  label_mapping:
  # Mapping of labels and task/activity in Physionet's dataset
    0:
    - - 3
      - 7
      - 11
    - - T1
    1:
    - - 3
      - 7
      - 11
    - - T2
```

## Experiments

**Batch norm vs layer norm for adaptability**

Use following values of parameters in `across_subject/params.yaml` under
block `across_subject_baseline` for hyperparamter tuning:

1. lr --> [0.01, 0.001]
2. batch_size --> [16, 32, 64]
3. norm --> [layer, batch]

For training run following command inside `across_subject` dir:

```shell
python3 baseline_train.py
```

For testing run following command inside `across_subject` dir:

```shell
python3 baseline_test.py
```

> NOTE:
>
> Running the script once, will create a model for one hyperparameter set. 

**Aross individual adaptability**

For MAML training, use following values of parameters in `across_subject/params.yaml` under
block `across_subject` for hyperparamter tuning:

1. adapt_lr --> [0.01, 0.001]
2. meta_lr --> [0.01, 0.001]
3. adapt_steps --> [5, 10]

For training run following command inside `across_subject` dir:

```shell
python3 train.py
```

For testing run following command inside `across_subject` dir:

```shell
python3 test.py
```

For transfer learning, use following values of parameters in `across_subject/params.yaml` under
block `across_subject_baseline` for hyperparamter tuning:

1. lr = [0.01, 0.001]
2. batch_size = [16, 32, 64]

For training run following command inside `across_subject` dir:

```shell
python3 baseline_train.py
```

For testing run following command inside `across_subject` dir:

```shell
python3 baseline_test.py
```

**Across task adaptability**

To evaluate the adaptability of the model on newer activities, we use the models traind during
`Aross individual adaptability`. We copy the model trained for one activity saved at
`across_subject/models` dir to `across_task/models` dir. We use the test scripts similar to ones in `across_subject` for testing.

To test on new activity, change the `label_mapping` in `across_task/params.yaml` for a new
activity and run the respective scripts in `across_task` dir.

For evaluation of MAML, use `across_task/test.py` and for transfer learning use `across_task/baseline_test.py`.

## Checkpoints

[Click here for checkpoints](https://iitgnacin-my.sharepoint.com/:f:/g/personal/22210006_iitgn_ac_in/Ej3LKfNH315FvCgHDBKIUtQBd4CJ2VvdM8x8YioWlD9YBA?e=FyTM0M)
