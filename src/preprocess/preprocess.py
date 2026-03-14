

def preprocess_raw(
        raw,
        sfreq,
        l_freq,
        h_freq
):
    """
    Resamples and applies band-pass filter to the raw EEG data.

    Parameters
    ----------
    raw : mne.RawArray
        The MNE Raw object.
    sfreq : int
        The resampling rate.
    l_freq : int
        Low-pass frequency for band-pass filter.
    h_freq  : int
        High-pass frequency for band-pass filter

    Returns
    -------
    mne.RawArray
        The filtered MNE Raw object.
    """
    raw = raw.copy()
    raw.resample(sfreq=sfreq)
    raw.filter(l_freq=l_freq, h_freq=h_freq)
    return raw