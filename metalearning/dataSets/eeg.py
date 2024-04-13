import numpy as np
from torch.utils.data import Dataset
from metalearning.utils import get_bci_2000

class BCI2000(Dataset):
    """Dataset class for EEG Motor Movement/Imagery Dataset[1]_ for MAML.

    .. [1] Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & \
    Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research \
    resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220.
    """
    def __init__(
            self, subjects: list, label_mapping: dict, relative_epoch_start_t: float,
            relative_epoch_end_t: float, crop_start: float, crop_end: float,
            filter_setting: dict, k_shot: int, k_qury: int, min_time_points: int
            ):
        """
        Initializing an object of BCI2000 class

        Parameters
        ----------
        subjects : list
            List of subject numbers for which the data needs to be fetched. The numbers
            should be within 1-109. Here, each subject is treated as a task from the 
            perspective of MAML.
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
        
        k_shot : int
            Number of samples in support set for each class
        k_qury : int
            Number of samples in query set for each class
        min_time_points : int
            Minimum number of data points in time domain
        """
        self.label_mapping: dict = label_mapping
        """Label mapping provided during object creation"""
        self.x_supports: list["np.ndarray"] = []
        """EEG signals of support set for all MAML tasks"""
        self.y_supports: list["np.ndarray"] = []
        """Labels of support set for all MAML tasks"""
        self.x_query: list["np.ndarray"] = []
        """EEG signals of query set for all MAML tasks"""
        self.y_query: list["np.ndarray"] = []
        """Labels of query set for all MAML tasks"""
        
        for subject in subjects:
            subject_eeg, subject_label = get_bci_2000(
                subject, label_mapping, relative_epoch_start_t, relative_epoch_end_t,
                crop_start, crop_end, filter_setting
                )
            
            if subject_eeg.shape[2] < min_time_points:
                print("Skipping. Less time points.")
                continue
            to_skip = False
            for k in label_mapping:
                if (subject_label == k).sum() < k_shot + k_qury:
                    to_skip = True
                    break
            if to_skip:
                to_skip = False
                print("Skipping. Less samples per class.")
                continue

            subject_x_s = []
            subject_y_s = []
            subject_x_q = []
            subject_y_q = []
            for k in label_mapping:
                class_index = np.where(subject_label == k)[0] # as np.where returns a tuple
                # if only condition is provided
                random_draws = np.random.choice(class_index, k_shot + k_qury, False)
                x, y = subject_eeg[random_draws], subject_label[random_draws]
                x_s, x_q = x[:k_shot], x[k_shot:]
                y_s, y_q = y[:k_shot], y[k_shot:]
                subject_x_s.append(x_s)
                subject_y_s.append(y_s)
                subject_x_q.append(x_q)
                subject_y_q.append(y_q)
            subject_x_s = np.vstack(subject_x_s)
            subject_y_s = np.hstack(subject_y_s)
            subject_x_q = np.vstack(subject_x_q)
            subject_y_q = np.hstack(subject_y_q)
            
            random_draws = np.random.choice(subject_x_s.shape[0], subject_x_s.shape[0], False)
            subject_x_s, subject_y_s = subject_x_s[random_draws], subject_y_s[random_draws]
            random_draws = np.random.choice(subject_x_q.shape[0], subject_x_q.shape[0], False)
            subject_x_q, subject_y_q = subject_x_q[random_draws], subject_y_q[random_draws]
            subject_x_s, subject_x_q = subject_x_s.astype(np.float32), subject_x_q.astype(np.float32)
        
            self.x_supports.append(subject_x_s)
            self.y_supports.append(subject_y_s)
            self.x_query.append(subject_x_q)
            self.y_query.append(subject_y_q)
    
    def __len__(self):
        return len(self.x_supports)
    
    def __getitem__(self, index):
        return (self.x_supports[index], self.y_supports[index], self.x_query[index], self.y_query[index])