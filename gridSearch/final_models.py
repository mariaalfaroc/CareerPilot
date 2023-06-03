import os
import warnings
# Disable warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

from data.load_data import read_and_preprocess_bin_data, read_and_preprocess_class_data

###### BINARY CLASSIFICATION:
# Using the best params from the binary classification grid search
def get_bin_model():
    return XGBClassifier(objective='binary:logistic',
                         n_estimators=300,
                         max_depth=4,
                         learning_rate=0.6,
                         n_jobs=-1,
                         tree_method='gpu_hist',
                         gpu_id=0)

###### MULTICLASS CLASSIFICATION:
# Using the best params from the multiclass classification grid search
def get_multi_model():
    return XGBClassifier(objective='multi:softmax',
                         n_estimators=150,
                         max_depth=3,
                         learning_rate=0.8,
                         n_jobs=-1,
                         tree_method='gpu_hist',
                         gpu_id=0)

####################################################################################################

MODELS = {
    'binary': get_bin_model(),
    'multiclass': get_multi_model()
}

def main():
    os.makedirs('final_results', exist_ok=True)
    os.makedirs('final_results/models', exist_ok=True)
    os.makedirs('final_results/logs', exist_ok=True)

    for name, model in MODELS.items():
        print(f'Training {name} model'.upper())

        # Load data
        if name == 'binary':
            X, y = read_and_preprocess_bin_data(path='data/05_mcvl_full_sample.csv')
        else:
            X, y = read_and_preprocess_class_data(path='data/05_mcvl_full_sample.csv')
            # Encode labels
            le = LabelEncoder()
            y = le.fit_transform(y)
            np.save('final_results/label_encoder.npy', le.classes_)

        # Create splits
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_splits = list(cv.split(X, y))
        np.save(f'final_results/{name}_cv_splits.npy', cv_splits)

        # Cross-validation
        acc_all = []
        f1_all = []
        f1_weighted_all = []
        for i, (train_index, test_index) in enumerate(cv.split(X, y)):
            print(f'\tFold {i}')
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # Compute sample weights
            train_sample_weight = compute_sample_weight('balanced', y_train)
            test_sample_weight = compute_sample_weight('balanced', y_test)
            # 1) Train
            model = model.fit(X_train, y_train, sample_weight=train_sample_weight)
            # 2) Predict
            y_pred = model.predict(X_test)
            # 3) Evaluate
            print('Random samples:')
            print('As output from the model:')
            print(f'\tTrue: {y_test[:5]}')
            print(f'\tPred: {y_pred[:5]}')
            # 3.1) Transform back to original labels
            if name == 'multiclass':
                y_test = le.inverse_transform(y_test)
                y_pred = le.inverse_transform(y_pred)
            print('As original labels:')
            print(f'\tTrue: {y_test[:5]}')
            print(f'\tPred: {y_pred[:5]}')
            # 3.2) Compute metrics
            # Binary. In the classification report:
            # - Accuracy -> row: accuracy, column: f1-score
            # - F1 -> row: positive class (1), column: f1-score
            # - F1 Weighted -> row: weighted avg, column: f1-score
            # Multiclass. In the classification report:
            # - Accuracy -> row: accuracy, column: f1-score
            # - F1 Weighted -> row: weighted avg, column: f1-score
            report = classification_report(y_test, y_pred, sample_weight=test_sample_weight, output_dict=True)
            # - Accuracy
            acc_all.append(report['accuracy'])
            print(f'\tAccuracy = {report["accuracy"]:.2f}')
            if name == 'binary':
                # - F1
                f1_all.append(report['1']['f1-score'])
                print(f'\tF1 = {report["1"]["f1-score"]:.2f}')
            # - F1 Weighted
            f1_weighted_all.append(report['weighted avg']['f1-score'])
            print(f'\tF1-Weighted = {report["weighted avg"]["f1-score"]:.2f}')
            # 3.3) Save results
            report = pd.DataFrame(report).transpose()
            report.to_csv(f'final_results/logs/{name}_model_fold_{i}.csv')
            # 3.4) Compute ROC AUC Curve
            if name == 'binary':
                # Positive class probabilities
                y_pred_probs = model.predict_proba(X_test)[:, 1]
                fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs, pos_label=1, sample_weight=test_sample_weight)
                auc = roc_auc_score(y_test, y_pred_probs, sample_weight=test_sample_weight)
                # Plotting the ROC curve
                plt.plot(fpr, tpr, label=f'Curva ROC (área = {auc:.2f})')
                plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('Tasa de Falsos Positivos')
                plt.ylabel('Tasa de Verdaderos Positivos')
                plt.title('Característica Operativa del Receptor (ROC)')
                plt.legend(loc='lower right')
                # Save the plot
                plt.savefig(f'final_results/plots/{name}_roc_curve_fold_{i}.png')
                plt.close()
            # 3.5) Compute confusion matrix
            cm = confusion_matrix(y_test, y_pred, sample_weight=test_sample_weight)
            sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues')
            plt.title('Matriz de Confusión')
            plt.xlabel('Predicho')
            plt.ylabel('Real') 
            plt.savefig(f'final_results/plots/{name}_confusion_matrix_fold_{i}.png')
            plt.close()
            # 3.6) Save model
            model.save_model(f'final_results/models/{name}_model_fold_{i}.json')
        # Print average metrics
        print(f'Average accuracy = {np.mean(acc_all):.2f}')
        if name == 'binary':
            print(f'Average f1 = {np.mean(f1_all):.2f}')
        print(f'Average f1-weighted = {np.mean(f1_weighted_all):.2f}')
        
        # Refit model on all data
        print('Refitting model on all data...', end=' ', flush=True)
        model = model.fit(X, y, sample_weight=compute_sample_weight('balanced', y))
        # Save model
        model.save_model(f'final_results/models/{name}_model.json')
        print('Done')
        print('-----------------------------------')

if __name__ == '__main__':
    main()
