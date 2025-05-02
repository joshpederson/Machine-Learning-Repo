import pandas as pd

# Load your data into a DataFrame
df = pd.read_csv('Loan_Defaulter.csv')

# Drop the specified columns
columns_to_drop = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
df = df.drop(columns=columns_to_drop)

# Convert specified columns to float64, handling errors
# columns_to_convert = [
#     'ID', 'LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 
#     'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 
#     'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
# ]
# for column in columns_to_convert:
#     df[column] = pd.to_numeric(df[column], errors='coerce')

# Drop rows with any NA values
df = df.dropna()

# Save the cleaned DataFrame to a new CSV file
df.to_csv('cleaned_file1.csv', index=False)

print("Data cleaned, columns converted, and saved to 'cleaned_file1.csv'")