# prepare_data.py
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
df = iris.frame
df.to_csv('data/iris.csv', index=False)
print("Initial IRIS dataset (150 samples) created.")
