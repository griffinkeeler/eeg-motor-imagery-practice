import numpy as np
from scipy.io import loadmat


class BCIDataset:
    """
    BCI Competition Dataset Class.

    This class can be used for to construct datasets from the BCI competition
     .mat files.

    Attributes
    ----------
    data : ndarray, shape (n_times, n_channels)
        The continuous EEG signals.
    markers : ndarray, shape (sample, cues)
        Position of cues in the EEG signals
    targets : ndarray, shape (n_targets)
        Vector of target classes (1 or 2)
    class_name : ndarray, shape (n_classes)
        The class names of the targets.
    """
    def __init__(self,
                 filepath):
        self.filepath = filepath
        self._load_mat()

    # TODO: Continue adding attributes (sfreq...)
    def _load_mat(self):
        file = loadmat(self.filepath)
        self.data = file["cnt"]
        self.markers = file["mrk"]["pos"][0, 0]
        targets = file["mrk"]["y"][0, 0].squeeze()
        self.targets = targets[~np.isnan(targets)].astype(int)
        self.class_name = file["mrk"]["className"][0, 0].squeeze()