import torch
from torch import nn
from torch.nn import functional as F
from mne import Epochs, pick_types, events_from_annotations
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
import numpy as np


def modelParametersInitializer(architecture: list) -> tuple:
    """Initialize parameters of a model being trained with MAML

    Parameters
    ----------
    architecture : list
        List of tuple defining the architecture

    Returns
    -------
    tuple
        Tuple containng parameters and batch norm parameter

    Raises
    ------
    NotImplementedError
        Raised when architecture component is not defined
    """    
    vars = nn.ParameterList()
    # running_mean and running_var
    vars_bn = nn.ParameterList()

    for _, (name, param) in enumerate(architecture):
        if name == 'conv2d':
            # [ch_out, ch_in, kernelsz, kernelsz]
            w = nn.Parameter(torch.ones((param[0], param[1]//param[6], param[2], param[3])))
            # gain=1 according to cbfin's implementation
            torch.nn.init.kaiming_normal_(w)
            vars.append(w)
            # [ch_out]
            vars.append(nn.Parameter(torch.zeros(param[0])))
        elif name == 'convt2d':
            # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
            w = nn.Parameter(torch.ones((param[0], param[1]//param[6], param[2], param[3])))
            # gain=1 according to cbfin's implementation
            torch.nn.init.kaiming_normal_(w)
            vars.append(w)
            # [ch_in, ch_out]
            vars.append(nn.Parameter(torch.zeros(param[1])))
        elif name == 'linear':
            # [ch_out, ch_in]
            w = nn.Parameter(torch.ones(*param))
            # gain=1 according to cbfinn's implementation
            torch.nn.init.kaiming_normal_(w)
            vars.append(w)
            # [ch_out]
            vars.append(nn.Parameter(torch.zeros(param[0])))
        elif name == 'bn':
            # [ch_out]
            w = nn.Parameter(torch.ones(param[0]))
            vars.append(w)
            # [ch_out]
            vars.append(nn.Parameter(torch.zeros(param[0])))

            # must set requires_grad=False
            running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
            running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
            vars_bn.extend([running_mean, running_var])
        elif name == 'ln':
            # [ch_out]
            w = nn.Parameter(torch.ones((param[0], param[1], param[2])))
            vars.append(w)
            # [ch_out]
            vars.append(nn.Parameter(torch.zeros((param[0], param[1], param[2]))))
        elif name in [
            'tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d', 'flatten', 'reshape',
            'leakyrelu', 'sigmoid', 'dropout', 'unsqueeze', 'elu'
        ]:
            continue
        else:
            raise NotImplementedError
    return vars, vars_bn

def get_bci_2000(
        subject: int, label_mapping: dict, relative_epoch_start_t: float,
        relative_epoch_end_t: float, crop_start: float, crop_end: float,
        filter_setting: dict
) -> tuple:
    """Returns recordings from EEG Motor Movement/Imagery Dataset[1]_ based on the 
    given parameters

    .. [1] Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & \
    Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research \
    resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220.

    Parameters
    ----------
    subject : int
        Subject number. Should be within 1-109
    label_mapping : dict
        The mapping of label and tasks. Each value in the dictionary should contain
        a list of tuples. First is the tuple of runs (1-14) and second is the tuple
        of desired annotation code (T0, T1, T2). Following example shows mapping of
        Task 1 with label 0 and Task 2 with label 1:

        ```python
        label_mapping = {
            0: [(3,7,11), ('T1', 'T2')],
            1: [(6,10,14), ('T1', 'T2')]
        }
        ```
        
    relative_epoch_start_t : float
        Start time of the epochs in seconds, relative to the time-locked event
    relative_epoch_end_t : float
        End time of the epochs in seconds, relative to the time-locked event
    crop_start : float
        Start time of selection in seconds
    crop_end : float
        End time of selection in seconds
    filter_setting : dict
        Parameters of Raw.filter() as key value pair (dict). Pass `None`, for
        no filtering. An example of `filter_setting` if given below:

        ```python
        filter_setting = {
        'l_freq': 7.0,
        'h_freq': 30.0,
        'fir_design': 'firwin',
        'skip_by_annotation': 'edge'
        }
        ```

    Returns
    -------
    tuple
        Tuple containing data and labels
    """    
    data = []
    label = []
    for k in label_mapping:
        runs = label_mapping[k][0]
        codes = label_mapping[k][1]
        raw_fnames = eegbci.load_data(subject, runs)
        raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
        eegbci.standardize(raw)  # set channel names
        montage = make_standard_montage("standard_1005")
        raw.set_montage(montage)
        if filter_setting:
            raw.filter(**filter_setting)
        
        event_id = dict()
        for code in codes:
            event_id[code] = k
        
        events, _ = events_from_annotations(raw, event_id=lambda x: event_id.get(x, None))
        print(f"Extracting runs: {runs} with annotation-label mapping as {_}")
        picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
        epochs = Epochs(
            raw,
            events,
            event_id,
            relative_epoch_start_t,
            relative_epoch_end_t,
            proj=True,
            picks=picks,
            baseline=None,
            preload=True,
        )
        epochs.crop(crop_start, crop_end)
        epoched_data = epochs.get_data()
        data.append(epoched_data)
        label.append([k] * epoched_data.shape[0])
    data = np.vstack(data)
    label = np.hstack(label)
    return data, label