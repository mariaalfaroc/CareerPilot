import os
import joblib
import warnings
# Disable warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

from data.load_data import read_and_preprocess_class_data

###### LOGISTIC REGRESION:
LOGIT_PARAM_GRID= {
    'logistic_classifier__penalty' : ['l2', None],
    'logistic_classifier__C' : np.logspace(-4, 4, 20),
    'logistic_classifier__solver' : ['newton-cg', 'sag', 'saga', 'lbfgs'],
    'logistic_classifier__max_iter' : [100, 1000, 2500, 5000]
}
def get_logit_clf():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('logistic_classifier', LogisticRegression(multi_class='multinomial', fit_intercept=True, n_jobs=-1, random_state=42))
    ])

###### XGBOOST:
XGB_PARAM_GRID = {
    'n_estimators': list(range(50, 400, 50)),
    'max_depth': list(range(1, 11)),
    'learning_rate': np.arange(0.1, 2.1, 0.1).tolist(),
}
def get_xgb_clf():
    return XGBClassifier(objective='multi:softmax', n_jobs=-1, tree_method='gpu_hist', gpu_id=0)

####################################################################################################

MODELS = {
    'logistic_regression': [get_logit_clf(), LOGIT_PARAM_GRID],
    'xgboost': [get_xgb_clf(), XGB_PARAM_GRID]
}

def run_hyperparameter_tuning(name, model, param_grid, X, y, cv):
    # Compute sample weights
    sample_weight = compute_sample_weight('balanced', y)
    # Define the grid search
    clf = RandomizedSearchCV(model,
                             param_distributions=param_grid,
                             n_iter=10,
                             refit='f1_weighted',
                             cv=cv,
                             verbose=2,
                             scoring=['accuracy', 'f1_weighted', 'roc_auc_ovr'],
                             return_train_score=True,
                             n_jobs=-1)
    # Fit the model
    if name == 'logistic_regression':
        best_clf = clf.fit(X, y, logistic_classifier__sample_weight=sample_weight)
    else:
        best_clf = clf.fit(X, y, sample_weight=sample_weight) 
    return best_clf

def main():
    os.makedirs('class_grid_results', exist_ok=True)
    os.makedirs('class_grid_results/models', exist_ok=True)
    os.makedirs('class_grid_results/logs', exist_ok=True)

    X, y = read_and_preprocess_class_data(path='data/04_mcvl_sample_model.csv')

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_splits = list(cv.split(X, y))
    np.save('class_grid_results/cv_splits.npy', cv_splits)

    le = LabelEncoder()
    y = le.fit_transform(y)
    np.save('class_grid_results/label_encoder.npy', le.classes_)

    for name, (clf, param_grid) in MODELS.items():
        print('Training', name)
        clf = run_hyperparameter_tuning(name, clf, param_grid, X, y, cv)    
        print('\tBest params', clf.best_params_)
        print('\tBest score', clf.best_score_)
        cv_results = pd.DataFrame.from_dict(clf.cv_results_)
        cv_results.to_csv(f'class_grid_results/logs/model_{name}_cv_results.csv')
        if name == 'xgboost':
            clf.best_estimator_.save_model(f'class_grid_results/models/model_{name}_best_estimator.json')
        else:
            joblib.dump(clf.best_estimator_, f'class_grid_results/models/model_{name}_best_estimator.joblib')

if __name__ == '__main__':
    main()