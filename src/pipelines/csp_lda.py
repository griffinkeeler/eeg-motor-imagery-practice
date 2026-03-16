import mne

from mne.decoding import CSP

from pathlib import Path
from src.dataset import BCIDataset
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Assemble CSP+LDA
    csp = CSP(
        n_components=4,
        log=True,
        norm_trace=False,
    )

    lda = LinearDiscriminantAnalysis()

    # Fit the CSP filters and transform the data
    csp.fit(X_train, y_train)
    X_train_features = csp.transform(X_train)
    X_test_features = csp.transform(X_test)

    # Classify targets using LDA
    lda.fit(X_train_features, y_train)
    y_pred = lda.predict(X_test_features)
    print("Accuracy:", accuracy_score(y_test, y_pred))



if __name__ == "__main__":
    main()