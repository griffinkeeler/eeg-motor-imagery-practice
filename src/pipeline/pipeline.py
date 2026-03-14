import mne

from mne.decoding import CSP, get_spatial_filter_from_estimator

from pathlib import Path
from src.dataset import BCIDataset
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda


def main():
    # Path to the data
    base_dir = Path(__file__).parents[2]
    data_dir = base_dir / "data" / "data_set_IVa_aa.mat"

    # Load the data
    dataset_iva_aa = BCIDataset(data_dir)
    raw = dataset_iva_aa.create_raw()
    events = dataset_iva_aa.create_events()
    event_dict = {"right hand": 1, "foot": 2}

    # Preprocess
    raw_processed = raw.copy()
    raw_processed, events_processed = raw_processed.resample(sfreq=200, events=events)
    raw_processed.filter(l_freq=8, h_freq=30)

    # Segment data into epochs
    epochs = mne.Epochs(
        raw_processed,
        events_processed,
        tmin=0.0,
        tmax=4.0,
        event_id=event_dict,
        preload=True,
        baseline=None,
    )

    # Obtain the training/testing data
    X = epochs.get_data()
    y = epochs.events[:, -1]

    csp = CSP(
        n_components=4,
        log=True,
        norm_trace=False,
    )

    # CSP+LDA Classifier
    clf = make_pipeline(csp, lda())


if __name__ == "__main__":
    main()