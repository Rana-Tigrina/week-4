import pandas as pd
import pytest

@pytest.fixture(scope="module")
def iris_data():
    return pd.read_csv("data/iris.csv")

def test_data_not_empty(iris_data):
    """Check that the dataset is not empty."""
    assert not iris_data.empty, "Dataset is empty!"

def test_no_missing_values(iris_data):
    """Check that there are no missing values."""
    assert iris_data.isnull().sum().sum() == 0, "Dataset contains missing values!"

def test_expected_columns(iris_data):
    """Check that expected columns exist."""
    expected_cols = {"sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)", "target"}
    assert set(iris_data.columns) == expected_cols, "Unexpected columns found!"

def test_target_values(iris_data):
    """Ensure target values are within expected range (0,1,2)."""
    unique_targets = iris_data["target"].unique()
    assert set(unique_targets).issubset({0, 1, 2}), f"Unexpected target values: {unique_targets}"

