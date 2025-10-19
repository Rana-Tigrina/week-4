import pytest
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

@pytest.fixture(scope="module")
def model_and_data():
    model = joblib.load("model.pkl")
    df = pd.read_csv("data/iris.csv")
    X = df.drop("target", axis=1)
    y = df["target"]
    return model, X, y

def test_model_accuracy(model_and_data):
    """Ensure model performs above threshold."""
    model, X, y = model_and_data
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    assert acc >= 0.8, f"Model accuracy too low: {acc}"

def test_model_predict_shape(model_and_data):
    """Check that model predictions match data length."""
    model, X, _ = model_and_data
    preds = model.predict(X)
    assert len(preds) == len(X), "Prediction length mismatch!"

