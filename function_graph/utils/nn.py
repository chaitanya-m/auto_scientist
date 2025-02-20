# utils/nn.py

def train_and_evaluate(model, dataset, train_ratio=0.5, epochs=1, verbose=0):
    """
    Splits 'dataset' into training and testing sets according to train_ratio,
    trains 'model' on the training set, and evaluates on the test set.

    Parameters
    ----------
    model : keras.Model
        The Keras model to train and evaluate.
    dataset : pd.DataFrame
        A DataFrame containing columns like 'feature_0', 'feature_1', ..., 'label', etc.
    train_ratio : float
        Fraction of the dataset to use for training (between 0 and 1).
    epochs : int
        Number of training epochs.
    verbose : int
        Verbosity mode for model.fit and model.predict.

    Returns
    -------
    float
        The accuracy on the test split.
    """
    split_idx = int(len(dataset) * train_ratio)
    train_df = dataset.iloc[:split_idx]
    test_df = dataset.iloc[split_idx:]

    train_features = train_df[[f"feature_{i}" for i in range(2)]].to_numpy(dtype=float)
    train_labels = train_df["label"].to_numpy(dtype=int)

    test_features = test_df[[f"feature_{i}" for i in range(2)]].to_numpy(dtype=float)
    test_labels = test_df["label"].to_numpy(dtype=int)

    model.fit(train_features, train_labels, epochs=epochs, verbose=verbose)
    predictions = model.predict(test_features, verbose=verbose)
    preds = (predictions.flatten() > 0.5).astype(int)
    accuracy = (preds == test_labels).mean()
    return accuracy
