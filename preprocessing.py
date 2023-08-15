import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler

def preprocess_data(df):
    # 1. Handling missing data (in this dataset there's none, but it's good to be prepared)
    df.fillna(df.median(), inplace=True)
    
    # 2. Feature Scaling
    std_scaler = StandardScaler()
    rob_scaler = RobustScaler()

    df['scaled_amount'] = std_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
    df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))

    df.drop(['Time', 'Amount'], axis=1, inplace=True)

    scaled_amount = df['scaled_amount']
    scaled_time = df['scaled_time']

    df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
    df.insert(0, 'scaled_amount', scaled_amount)
    df.insert(1, 'scaled_time', scaled_time)

    # 3. Data Splitting
    y = df['Class']
    X = df.drop('Class', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = pd.read_csv('creditcard.csv')
    X_train, X_test, y_train, y_test = preprocess_data(df)

    X_train.to_csv('processed_X_train.csv', index=False)
    X_test.to_csv('processed_X_test.csv', index=False)
    y_train.to_csv('processed_y_train.csv', index=False, header=False)
    y_test.to_csv('processed_y_test.csv', index=False, header=False)

    print(X_train.head())
    print(y_train.head())
