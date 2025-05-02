import pandas as pd
import numpy as np

cleaned = pd.read_csv("dataGaia2.csv")

# Drop columns that only contain strings
for col in cleaned.columns:
    if cleaned[col].dtype == 'object':
        cleaned.drop(columns=[col], inplace=True)

# Drop the specific column named 'Flags-HS'
if 'Flags-HS' in cleaned.columns:
    cleaned.drop(columns=['Flags-HS'], inplace=True)

cleaned.dropna(inplace=True)

cleaned.to_csv("cleaned_data.csv")

print("Data cleaned and saved to cleaned_data.csv")