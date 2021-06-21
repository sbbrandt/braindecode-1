"""Process a big data EEG resource (TUH EEG Corpus)
===================================================

In this example, we showcase usage of the Temple University Hospital EEG Corpus
(https://www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml#c_tueg)
including simple preprocessing steps as well as cutting of compute windows.
"""

# Author: Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD (3-clause)

import os

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import mne

from braindecode.datasets import TUH
from braindecode.preprocessing.preprocess import preprocess, Preprocessor
from braindecode.preprocessing.windowers import create_fixed_length_windows
from braindecode.datautil.serialization import (
    save_concat_dataset, load_concat_dataset)

mne.set_log_level('ERROR')  # avoid messages everytime a window is extracted


###############################################################################
# We start by creating a TUH dataset. First, the class generates a description
# of the recordings in `TUH_PATH` (which is later accessible as
# `tuh.description`) without actually touching the files. This will parse
# information from file paths such as patient id, recording data, etc and should
# be really fast. Afterwards, the files are sorted chronologically by year,
# month, day, patient id, recording session and segment.
# In the following, a subset of the description corresponding to `recording_ids`
# is used.
# Afterwards, the files will be iterated a second time, slower than before.
# The files are now actually touched. Additional information about subjects
# like age and gender are parsed directly from the EDF file header. If existent,
# the physician report is added to the description. Furthermore, the recordings
# are read with `mne.io.read_raw_edf` with `preload=False`. Finally, we will get
# a `BaseConcatDataset` of `BaseDatasets` each holding a single
# `nme.io.Raw` which is fully compatible with other braindecode functionalities.
TUH_PATH = '/home/lukas/Downloads/tuh_eeg_sample/'
tuh = TUH(
    path=TUH_PATH,
    recording_ids=None,
    target_name=None,
    preload=False,
    add_physician_reports=True,
)


###############################################################################
# We can easily create descriptive statistics using the description `DataFrame`,
# for example an age histogram split by gender of patients.
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
genders = tuh.description.gender.unique()
x = [tuh.description.age[tuh.description.gender == g] for g in genders]
ax.hist(
    x=x,
    stacked=True,
    bins=np.arange(100, dtype=int),
    alpha=.5,
)
ax.legend(genders)


###############################################################################
# Next, we will perform some preprocessing steps. First, we will do some
# selection of available recordings based on the duration. We will select those
# recordings, that have at least five minutes duration. Data is not loaded here.


def select_by_duration(ds, tmin=0, tmax=None):
    # determine length of the recordings and select based on tmin and tmax
    duration = ds.description.n_samples / ds.description.sfreq
    duration = duration[duration >= tmin]
    if tmax is None:
        tmax = np.inf
    duration = duration[duration <= tmax]
    split_ids = list(duration.index)
    splits = ds.split(split_ids)
    split = splits['0']
    return split


tmin = 5 * 60
tmax = None
tuh = select_by_duration(tuh, tmin, tmax)


###############################################################################
# Next, we will discard all recordings that have an incomplete channel
# configuration (wrt the channels that we are interested in, i.e. the 21
# channels of the international 10-20-placement). The dataset is subdivided into
# recordings with 'le' and 'ar' reference which we will have to consider. Data
# is not loaded here.

short_ch_names = sorted([
    'A1', 'A2',
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ'])
ar_ch_names = sorted([
    'EEG A1-REF', 'EEG A2-REF',
    'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF',
    'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF',
    'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF',
    'EEG T6-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF'])
le_ch_names = sorted([
    'EEG A1-LE', 'EEG A2-LE',
    'EEG FP1-LE', 'EEG FP2-LE', 'EEG F3-LE', 'EEG F4-LE', 'EEG C3-LE',
    'EEG C4-LE', 'EEG P3-LE', 'EEG P4-LE', 'EEG O1-LE', 'EEG O2-LE',
    'EEG F7-LE', 'EEG F8-LE', 'EEG T3-LE', 'EEG T4-LE', 'EEG T5-LE',
    'EEG T6-LE', 'EEG FZ-LE', 'EEG CZ-LE', 'EEG PZ-LE'])
assert len(short_ch_names) == len(ar_ch_names) == len(le_ch_names)
ar_ch_mapping = {ch_name: short_ch_name for ch_name, short_ch_name in zip(
    ar_ch_names, short_ch_names)}
le_ch_mapping = {ch_name: short_ch_name for ch_name, short_ch_name in zip(
    le_ch_names, short_ch_names)}
ch_mapping = {'ar': ar_ch_mapping, 'le': le_ch_mapping}


def select_by_channels(ds, ch_mapping):
    split_ids = []
    for i, d in enumerate(ds.datasets):
        seta = set(ch_mapping[d.description.reference].keys())
        setb = set(d.raw.ch_names)
        if seta.issubset(setb):
            split_ids.append(i)
    return ds.split(split_ids)['0']


tuh = select_by_channels(tuh, ch_mapping)


###############################################################################
# Next, we will chain several preprocessing steps that are realized through
# `mne`. Data will be loaded by the first preprocessor that has a mention of it
# in brackets:
# - crop the recordings to a region of interest
# - re-reference all recordings to 'ar' (requires load)
# - rename channels to short channel names
# - pick channels of interest
# - scale signals to microvolts (requires load)
# - resample recordings to a common frequency (requires load)
# - create compute windows

def custom_rename_channels(raw, mapping):
    # rename channels which are dependent on referencing:
    # le: EEG 01-LE, ar: EEG 01-REF
    # mne fails if the mapping contains channels as keys that are not present
    # in the raw
    reference = raw.ch_names[0].split('-')[-1].lower()
    assert reference in ['le', 'ref'], 'unexpected referencing'
    reference = 'le' if reference == 'le' else 'ar'
    raw.rename_channels(mapping[reference])


def custom_crop(raw, tmin=0.0, tmax=None, include_tmax=True):
    # crop recordings to tmin – tmax. can be incomplete if recording
    # has lower duration than tmax
    # by default mne fails if tmax is bigger than duration
    tmax = min((raw.n_times - 1) / raw.info['sfreq'], tmax)
    raw.crop(tmin=tmin, tmax=tmax, include_tmax=include_tmax)


tmin = 1 * 60
tmax = 6 * 60
sfreq = 100

preprocessors = [
    Preprocessor(custom_crop, tmin=tmin, tmax=tmax, include_tmax=False,
                 apply_on_array=False),
    Preprocessor('set_eeg_reference', ref_channels='average', ch_type='eeg'),
    Preprocessor(custom_rename_channels, mapping=ch_mapping,
                 apply_on_array=False),
    Preprocessor('pick_channels', ch_names=short_ch_names, ordered=True),
    Preprocessor(lambda x: x * 1e6),
    Preprocessor('resample', sfreq=sfreq),
]


###############################################################################
# The preprocessing loop works as follows. For every recording, we apply the
# preprocessors as defined above. Then, we update the description of the rec,
# since we have altered the duration, the reference, and the sampling frequency.
# Afterwards, we split the continuous signals into compute windows. We store
# each recording to a unique subdirectory that is named corresponding to the
# rec id. To save memory, after windowing and storing, we delete the raw
# dataset and the windows dataset, respectively.
window_size_samples = 1000
window_stride_samples = 1000
create_compute_windows = True

out_i = 0
errors = []
OUT_PATH = './tuh_sample/'
tuh_splits = tuh.split([[i] for i in range(len(tuh.datasets))])
for rec_i, tuh_subset in tuh_splits.items():
    # implement preprocess for BaseDatasets? Would remove necessity
    # to split above
    preprocess(tuh_subset, preprocessors)

    # update description of the recording(s)
    tuh_subset.description.sfreq = len(tuh_subset.datasets) * [sfreq]
    tuh_subset.description.reference = len(tuh_subset.datasets) * ['ar']
    tuh_subset.description.n_samples = [len(d) for d in tuh_subset.datasets]

    if create_compute_windows:
        # generate compute windows here and store them to disk
        tuh_windows = create_fixed_length_windows(
            tuh_subset,
            start_offset_samples=0,
            stop_offset_samples=None,
            window_size_samples=window_size_samples,
            window_stride_samples=window_stride_samples,
            drop_last_window=False
        )
        # save memory by deleting raw recording
        del tuh_subset
        # store the number of windows required for loading later on
        tuh_windows.description["n_windows"] = [len(d) for d in
                                                tuh_windows.datasets]

        # create one directory for every recording
        rec_path = os.path.join(OUT_PATH, str(rec_i))
        if not os.path.exists(rec_path):
            os.makedirs(rec_path)
        save_concat_dataset(rec_path, tuh_windows)
        out_i += 1
        # save memory by catching epoched recording
        del tuh_windows
    else:
        # store raws to disk for option of using different compute window
        # sizes
        pass


###############################################################################
# We load the preprocessed data again in a lazy fashion (`preload=False`). It is
# now ready to be used for model training.

tuh_loaded = load_concat_dataset('./tuh_sample/', preload=False)
