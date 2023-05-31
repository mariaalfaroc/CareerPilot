import os
import warnings
# Disable warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

from data.load_data import read_and_preprocess_class_data

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
    'xgboost': [get_xgb_clf(), XGB_PARAM_GRID]
}

def run_hyperparameter_tuning(model, param_grid, X, y, cv):
    clf = RandomizedSearchCV(model,
                             param_distributions=param_grid,
                             n_iter=10,
                             refit=True,
                             cv=cv,
                             verbose=2,
                             scoring='accuracy',
                             return_train_score=True,
                             n_jobs=-1)
    # Fit the model
    best_clf = clf.fit(X, y)
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

    for model, (clf, param_grid) in MODELS.items():
        print('Training', model)
        clf = run_hyperparameter_tuning(clf, param_grid, X, y, cv)    
        print('\tBest params', clf.best_params_)
        print('\tBest score', clf.best_score_)
        cv_results = pd.DataFrame.from_dict(clf.cv_results_)
        cv_results.to_csv(f'class_grid_results/logs/model_{model}_cv_results.csv')
        clf.best_estimator_.save_model(f'class_grid_results/models/model_{model}_best_estimator.json')

if __name__ == '__main__':
    main()