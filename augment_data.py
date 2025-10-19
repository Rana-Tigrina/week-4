# augment_data.py
import pandas as pd

df = pd.read_csv('data/iris.csv')
new_samples = df.tail(50).copy() # Simple duplication for demonstration
df_augmented = pd.concat([df, new_samples], ignore_index=True)
df_augmented.to_csv('data/iris.csv', index=False)
print(f"Data augmented. New total samples: {len(df_augmented)}")
