import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Load the dataset
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Preprocess the dataset
@st.cache_data
def preprocess_data(data):
    # Handle missing values
    data = data.dropna(subset=['treatment'])  # Drop rows with missing target
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column].fillna("Unknown", inplace=True)  # Replace missing values in categorical columns
        else:
            data[column].fillna(data[column].median(), inplace=True)  # Replace missing values in numerical columns

    # Encode categorical variables
    label_encoders = {}
    for column in data.columns:
        if data[column].dtype == 'object':
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])
            label_encoders[column] = le

    # Separate features and target
    X = data.drop(columns=['treatment'])
    y = data['treatment']

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)

    # Standardize numerical features
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X, y, label_encoders

# Hyperparameter tuning
@st.cache_resource
def tune_hyperparameters(X, y):
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }
    grid_search = GridSearchCV(estimator=XGBClassifier(eval_metric='logloss', random_state=42),
                               param_grid=param_grid,
                               scoring='accuracy', cv=5)
    grid_search.fit(X, y)
    return grid_search.best_estimator_

# Train the model
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = tune_hyperparameters(X_train, y_train)
    model.fit(X_train, y_train)
    return model, X_train.columns

# Streamlit App
st.title("Enhanced Mental Health Analysis Tool")
st.write("Analyze the likelihood of requiring mental health treatment using advanced ML techniques.")

# File path for your dataset
file_path = 'Mental_Health_Dataset.csv'

# Load and preprocess the data
data = load_data(file_path)
X, y, label_encoders = preprocess_data(data)
model, feature_names = train_model(X, y)

# Sidebar for user inputs
st.sidebar.header("Input Parameters")
input_data = {}
for feature in feature_names:
    options = data[feature].dropna().unique()  # Exclude NaN from dropdown options
    input_data[feature] = st.sidebar.selectbox(f"Select value for {feature}", options)

# Convert user input to DataFrame
input_df = pd.DataFrame([input_data])

# Encode input data
for feature, encoder in label_encoders.items():
    if feature in input_df.columns:
        input_df[feature] = encoder.transform(input_df[feature])

# Make predictions
if st.button("Analyze"):
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

    if prediction == 1:
        st.error(f"The analysis suggests a higher likelihood of requiring mental health treatment. "
                 f"Confidence: {prediction_proba[1] * 100:.2f}%")
    else:
        st.success(f"The analysis suggests a lower likelihood of requiring mental health treatment. "
                   f"Confidence: {prediction_proba[0] * 100:.2f}%")
