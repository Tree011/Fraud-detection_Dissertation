import pandas as pd
import numpy as np

# Number of transactions 
n = 10000

# Percentage of fraudulent transactions
fraud_percent = 0.01 

# Compute number of fraudulent and legitimate transactions
n_fraud = int(n * fraud_percent)  
n_legit = n - n_fraud

# Generate synthetic data
scaled_time = np.random.randn(n)
scaled_amount = np.random.randn(n)

# Generate 28 random features 
V_cols = [np.random.randn(n) for _ in range(28)]

# Generate class labels 
labels = np.array([1]*n_fraud + [0]*n_legit)
np.random.shuffle(labels)

# Assemble into dataframe
data = [scaled_time, scaled_amount] + V_cols + [labels]
df_synthetic = pd.DataFrame(data).T

# Add column names
df_synthetic.columns = ['scaled_time', 'scaled_amount'] + [f'V{i}' for i in range(1, 29)] + ['Class']

# Save to CSV
df_synthetic.to_csv('synthetic_credit_fraud.csv', index=False)