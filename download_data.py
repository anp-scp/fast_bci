"""Script to download EEG data"""

from mne.datasets import eegbci

for subject in range(1,110):
    # download data and forget. If data already exists, then it will do nothing
    eegbci.load_data(subject, [3,4,5,6,7,8,9,10,11,12,13,14])

