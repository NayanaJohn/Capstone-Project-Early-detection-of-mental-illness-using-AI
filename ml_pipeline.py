import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

class MentalHealthModel:
    def __init__(self):
        self.pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('selector', SelectKBest(chi2, k=16)),
            ('classifier', CatBoostClassifier(iterations=100, random_seed=42, verbose=0))
        ])
        self.selected_feature_names = None  # Store feature names

    def preprocess_mental_health_data(self, input_df):
        processed_df = input_df.copy()
        columns_to_encode = ['Gender', 'Ethnicity', 'Employment_Status', 'Current_Mental_Health_Challenges', 'Family_History_Mental_Health']
        processed_df = pd.get_dummies(processed_df, columns=columns_to_encode, drop_first=False, dtype=int)

        mappings = {
            'Age': {'25-34': 3, '18-24': 4, '35-44': 2, '45-54': 1, '55 and above': 5},
            'Workload_Manageability': {'Often': 4, 'Rarely': 3, 'Sometimes': 1, 'Never': 2, 'Always': 5, 'Not applicable': -1},
            'Burnout_Frequency': {'Often': 5, 'Rarely': 1, 'Sometimes': 2, 'Never': 4, 'Always': 3, 'Not applicable': -1},
            'Sleep_Hours': {'6 - 7 hours': 3, '8 - 9 hours': 3, '4 - 5 hours': 4, 'Less than 4 hours': 5, 'More than 9 hours': 1, 'Unknown': -1},
            'Physical_Activity_Hours': {'Less than 1 hour': 2, '3 - 4 hours': 4, '1 - 2 hours': 3, '5 - 6 hours': 5, 'More than 6 hours': 1},
            'Relaxation_Challenge': {'Sometimes': 4, 'Rarely': 1, 'Often': 2, 'Always': 5, 'Never': 3, 'Unknown': -1, 'Not Applicable': -1},
            'Mood_Swings': {'Weekly': 4, 'Rarely': 2, 'Monthly': 3, 'Daily': 5, 'Never': 1, 'Unknown': -1},
            'Work_Life_Balance_Satisfaction': {'Satisfied': 1, 'Neutral': 2, 'Dissatisfied': 4, 'Very satisfied': 3, 'Very dissatisfied': 5},
            'Help_Seeking_Behavior': {'Very likely': 5, 'Very unlikely': 2, 'Likely': 4, 'Neutral': 1, 'Unlikely': 3, 'Unknown': -1},
            'Relaxation_Activities': {'Daily': 5, 'Weekly': 1, 'Never': 2, 'Rarely': 3, 'Monthly': 4, 'Unknown': -1},
            'Days_Off_Mental_Health': {'No': 0, 'Yes': 1, 'Not applicable': -1, 'Unknown': -1},
            'Appetite_Changes': {'Yes': 1, 'No': 0, 'Unknown': -1, 'Not Applicable': -1},
            'Mental_Health_Diagnosis': {'Yes': 1, 'No': 0}
        }
        for feature, mapping in mappings.items():
            if feature in processed_df.columns:
                processed_df[feature] = processed_df[feature].map(mapping).fillna(-1)

        for col in ['Alcohol_Consumption', 'Smoking', 'Caffeine_Intake', 'Sugary_Snacks']:
            freq_map = processed_df[col].value_counts(normalize=True).to_dict()
            processed_df[col] = processed_df[col].map(freq_map)

        return processed_df

    def handle_imbalance(self, X_train, y_train):
        train_data = pd.concat([X_train, y_train], axis=1)
        majority_class = train_data[train_data['Mental_Health_Diagnosis'] == 0]
        minority_class = train_data[train_data['Mental_Health_Diagnosis'] == 1]
        majority_downsampled = resample(
            majority_class,
            replace=True,
            n_samples=4842,
            random_state=42
        )
        downsampled_train_data = pd.concat([majority_downsampled, minority_class])
        X_train_downsampled = downsampled_train_data.drop('Mental_Health_Diagnosis', axis=1)
        y_train_downsampled = downsampled_train_data['Mental_Health_Diagnosis']
        return X_train_downsampled, y_train_downsampled

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        # Get selected feature names after fitting the model
        self.selected_feature_names = X.columns[self.pipeline.named_steps['selector'].get_support()].tolist()

    def predict(self, X):
        X = self.align_features(X)
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        X = self.align_features(X)
        return self.pipeline.predict_proba(X)[:, 1]

    def align_features(self, X):
        if self.selected_feature_names is not None:
            for col in self.selected_feature_names:
                if col not in X.columns:
                    X[col] = 0
            X = X[self.selected_feature_names]
        return X

    def save_model(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)

