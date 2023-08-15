import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('creditcard.csv')

# Display the first few rows
print(data.head())

# Basic statistics
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Visualize class distribution
plt.figure(figsize=(8, 6))
sns.countplot(data['Class'])
plt.title('Class Distribution (0: No Fraud, 1: Fraud)')
plt.show()
plt.savefig("class_distribution.jpg")


# Visualize transaction amounts for fraudulent and non-fraudulent transactions
frauds = data[data['Class'] == 1]
non_frauds = data[data['Class'] == 0]

plt.figure(figsize=(10, 6))
plt.hist([frauds['Amount'], non_frauds['Amount']], bins=50, alpha=0.5, label=['Fraud', 'No Fraud'])
plt.xlabel('Transaction Amount ($)')
plt.ylabel('Number of Transactions')
plt.title('Transaction Amount vs. Number of Transactions')
plt.legend(loc='upper right')
plt.show()
plt.savefig("TransactionamtVNooftrans.jpg")


# Visualize transaction time distribution
plt.figure(figsize=(10, 6))
plt.hist([frauds['Time'], non_frauds['Time']], bins=50, alpha=0.5, label=['Fraud', 'No Fraud'])
plt.xlabel('Time (Seconds)')
plt.ylabel('Number of Transactions')
plt.title('Transaction Time vs. Number of Transactions')
plt.legend(loc='upper right')
plt.show()
plt.savefig("TransTimeVNooftrans.jpg")

