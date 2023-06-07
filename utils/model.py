import joblib
import pickle

from xgboost import XGBClassifier

def load_model(path):
    """Load a model from a file."""
    # Joblib
    if path.endswith('.joblib'):
        return joblib.load(path)
    # XGBoost
    elif path.endswith('.json'):
        model = XGBClassifier()
        model.load_model(path)
        return model
    # Pickle
    elif path.endswith('.pkl'):
        with open(path, 'rb') as file:
            return pickle.load(file)