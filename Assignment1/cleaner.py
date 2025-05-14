import pandas as pd
import numpy as np

df = pd.read_csv("powerplants.csv")
# Drop rows with NaN values in specified columns
df_cleaned = df.dropna(subset=["capacity in MW", "generation_gwh_2021", 'estimated_generation_gwh_2021','latitude'])

print(df_cleaned['capacity in MW'].isna().sum())
print(df_cleaned['generation_gwh_2021'].isna().sum())
print(df_cleaned['estimated_generation_gwh_2021'].isna().sum())
#print(df_cleaned['latitude'].isna().sum())
#print(df_cleaned['longitude'].isna().sum())

print(df_cleaned.count())
# Save the cleaned DataFrame to a new CSV file
df_cleaned.to_csv('cleaned_powerplant_data1.csv', index=False)