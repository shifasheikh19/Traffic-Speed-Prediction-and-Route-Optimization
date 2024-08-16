import pandas as pd
import numpy as np

# Load the merged dataset
merged_data = pd.read_csv('merged_data.csv')

# Display initial statistics
print("Initial statistics:")
print(merged_data.describe())

# Handling missing values
merged_data.ffill(inplace=True)

# Display statistics after filling missing values
print("\nStatistics after filling missing values:")
print(merged_data.describe())

# Handle outliers by capping values
numeric_cols = ['temp', 'feels_like', 'freeFlowSpeed', 'currentSpeed']
for col in numeric_cols:
    q_low = merged_data[col].quantile(0.01)
    q_high = merged_data[col].quantile(0.99)
    merged_data = merged_data[(merged_data[col] >= q_low) & (merged_data[col] <= q_high)]

# Display statistics after handling outliers
print("\nStatistics after handling outliers:")
print(merged_data.describe())

# Ensure data consistency
merged_data['weather_main'] = merged_data['weather_main'].astype('category')

# One-hot encoding for the categorical 'weather_main' feature
merged_data = pd.get_dummies(merged_data, columns=['weather_main'])

# Interaction features
merged_data['temp_freeFlowSpeed'] = merged_data['temp'] * merged_data['freeFlowSpeed']
merged_data['feels_like_freeFlowSpeed'] = merged_data['feels_like'] * merged_data['freeFlowSpeed']

# Define the updated features list including the one-hot encoded columns and new features
features = ['freeFlowSpeed', 'temp', 'feels_like', 'temp_freeFlowSpeed', 'feels_like_freeFlowSpeed'] + [col for col in merged_data.columns if col.startswith('weather_main_')]

# Target variable
target = 'currentSpeed'

# Save cleaned and processed data for further steps
merged_data.to_csv('cleaned_merged_data_fixed.csv', index=False)

print("Data cleaning and feature engineering completed successfully.")
