import pandas as pd
import numpy as np
import joblib
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import itertools

# Create a directory to save the model and preprocessors
if not os.path.exists('models'):
    os.makedirs('models')

print("Starting model training and setup...")

# Load the datasets. Assuming 'Train_data.csv' and 'Test_data.csv' are in a 'data' folder.
try:
    train_df = pd.read_csv('data/Train_data.csv')
    test_df = pd.read_csv('data/Test_data.csv')
except FileNotFoundError:
    print("Error: Train_data.csv or Test_data.csv not found.")
    print("Please place the CSV files in a 'data' directory in the same folder as this script.")
    exit()

# Save the original list of column names, excluding 'class'
original_columns = list(train_df.drop('class', axis=1).columns)
joblib.dump(original_columns, 'models/original_columns.joblib')

# 1. Label Encoding for all categorical columns
# We fit the LabelEncoder on combined unique values from both train and test sets to avoid 'unseen labels' error
def le(train_df, test_df):
    encoders = {}
    
    # Identify all categorical columns across both dataframes
    all_cols = list(set(train_df.columns) | set(test_df.columns))
    categorical_cols = [col for col in all_cols if train_df[col].dtype == 'object' or (col in test_df.columns and test_df[col].dtype == 'object')]

    for col in categorical_cols:
        le = LabelEncoder()
        
        train_unique = train_df[col].astype(str).unique() if col in train_df.columns else []
        test_unique = test_df[col].astype(str).unique() if col in test_df.columns else []
        
        combined_unique_values = np.unique(np.concatenate((train_unique, test_unique)))
        
        le.fit(combined_unique_values)
        encoders[col] = le
        
        if col in train_df.columns:
            train_df[col] = le.transform(train_df[col].astype(str))
        
        if col in test_df.columns:
            test_df[col] = le.transform(test_df[col].astype(str))
            
    return train_df, test_df, encoders

print("Applying label encoding...")
train_df, test_df, label_encoders = le(train_df, test_df)
joblib.dump(label_encoders, 'models/label_encoders.joblib')

# 2. Drop 'num_outbound_cmds' as it has no variance
print("Dropping 'num_outbound_cmds' column...")
if 'num_outbound_cmds' in train_df.columns:
    train_df.drop(['num_outbound_cmds'], axis=1, inplace=True)
if 'num_outbound_cmds' in test_df.columns:
    test_df.drop(['num_outbound_cmds'], axis=1, inplace=True)

# Separate features and target
X_train = train_df.drop(['class'], axis=1)
Y_train = train_df['class']

# 3. Feature Selection with RFE
print("Performing feature selection...")
rfc = RandomForestClassifier(random_state=42)
rfe = RFE(rfc, n_features_to_select=10)
rfe = rfe.fit(X_train, Y_train)

feature_map = [(i, v) for i, v in itertools.zip_longest(rfe.get_support(), X_train.columns)]
selected_features = [v for i, v in feature_map if i==True]
joblib.dump(selected_features, 'models/selected_features.joblib')
print(f"Selected features: {selected_features}")

# Filter the training data to only include the selected features
X_train = X_train[selected_features]

# 4. Standard Scaling
print("Applying standard scaling...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, 'models/scaler.joblib')

# 5. Train the KNN model
print("Training the K-Nearest Neighbors (KNN) model...")
KNN_model = KNeighborsClassifier(n_neighbors=3)
KNN_model.fit(X_train_scaled, Y_train)

# 6. Save the trained model
print("Saving the trained model...")
joblib.dump(KNN_model, 'models/knn_model.joblib')

print("Model setup complete. You can now run 'app.py'.")
