
import os
import sys
from dataclasses import dataclass
from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier
)

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, FunctionTransformer
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from scikeras.wrappers import KerasClassifier

from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging

from data_transformation import DataTransformer

from src.utils import *
from src.config import *

import optuna
from optuna import Trial
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances, plot_slice

import dill

class ModelTuner:
    def __init__(self, n_trials, preprocessor, cv):
        self.n_trials = n_trials
        self.preprocessor = preprocessor
        self.cv = cv
        self.best_params = {}
        self.best_score = 0
        self.study = None

    def objective(self, trial:Trial, model_name, smote=True, class_weight='balanced'):
        if model_name == 'XGBoost':
            params = {
                'objective': 'multi:softprob',
                'random_state': RANDOM_SEED,
                'tree_method': 'auto',
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 2, 10),
                'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 5.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.2, 0.8, step=0.1),
                'gamma': trial.suggest_float('gamma', 0, 9)
            }
        elif model_name == 'CatBoost':
            params = {
                'random_state': RANDOM_SEED,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 30, 1500, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.5,2,log=True),
                'rsm': trial.suggest_float('rsm', 0.5, 1.0, log=True),
            }
        elif model_name == 'Random Forest':
            params = {
                'random_state':RANDOM_SEED,
                'n_estimators': trial.suggest_int('n_estimators', 50, 500, log=True),
                'max_depth': trial.suggest_int('max_depth', 10, 100),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20)
            }
        # params = {
        #     'objective': 'multi:softprob',
        #     'random_state': RANDOM_SEED,
        #     'learning_rate': 0.09945002602812084, 
        #     'max_depth': 5, 
        #     'min_child_weight': 0.2784016726023859,
        #     'subsample': 0.4, 
        #     'gamma': 1.8685907147813174
        # }

        predict_pipeline = Pipeline(steps=[
                ('Preprocessor', self.preprocessor),
                ('Fix Names', FunctionTransformer(clean_column_names)),
                ('Add Features', FunctionTransformer(feature_adder)),
            ])
        
        if smote:
            # sample_strat = {
        #     0: trial.suggest_int('A', get_sample_ratio('A', .82), get_sample_ratio('C', .82), log=True),
        #     1: get_sample_ratio('C', .82),
        #     2: trial.suggest_int('D', get_sample_ratio('D', .82), get_sample_ratio('C', .82), log=True),
        #     3: trial.suggest_int('L', get_sample_ratio('L', .82), get_sample_ratio('C', .82), log=True),
        #     4: trial.suggest_int('Q', get_sample_ratio('Q', .82), get_sample_ratio('C', .82), log=True),
        #     5: get_sample_ratio('S', .82),
        #     6: trial.suggest_int('V', get_sample_ratio('V', .82), get_sample_ratio('C', .82), log=True),
        #     7: trial.suggest_int('X', get_sample_ratio('X', .82), get_sample_ratio('C', .82), log=True)
        # }
            predict_pipeline.steps.append(('SMOTE', SMOTE(sampling_strategy='auto', k_neighbors=trial.suggest_int('K', 5, 30))))
        
        if model_name == 'XGBoost':
            clf = XGBClassifier(**params)
        elif model_name == 'CatBoost':
            clf = CatBoostClassifier(**params)
        elif model_name == 'Random Forest':
            clf = RandomForestClassifier(**params)

        predict_pipeline.steps.append((model_name, clf))

        train_df = get_train_df()

        X, y = get_X_and_y(train_df, TARGET)

        y = label_encode(y)

        

        # class_weight_map = {
        #     0: trial.suggest_float('A', 0.35, 25, log=True),
        #     1: trial.suggest_float('C', 0.35, 25, log=True),
        #     2: trial.suggest_float('D', 0.35, 25, log=True),
        #     3: trial.suggest_float('L', 0.35, 25, log=True),
        #     4: trial.suggest_float('Q', 0.35, 25, log=True),
        #     5: trial.suggest_float('S', 0.35, 25, log=True),
        #     6: trial.suggest_float('V', 0.35, 25, log=True),
        #     7: trial.suggest_float('X', 0.35, 25, log=True)
        # }
        
        scores = []

        for train_index, test_index in self.cv.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            if class_weight:
                class_weights = compute_sample_weight(
                class_weight='balanced',
                y=y_train
            )
                predict_pipeline.fit(X_train, y_train, **{f'{model_name}__sample_weight': class_weights})
            else:
                predict_pipeline.fit(X_train, y_train)
            y_pred_proba = predict_pipeline.predict_proba(X_test)
            scores.append(roc_auc_score(y_test, y_pred_proba, multi_class='ovr'))

        return np.mean(scores)

    def cv_matrix(self, model_name, class_weight='balanced', params = {}, 
                  smote=False, smote_sampling_strategy='auto', under_sample=False, display=True):
        train_df = get_train_df()
        X, y = get_X_and_y(train_df, TARGET)
        y = label_encode(y)

        predict_pipeline = Pipeline(steps=[
                ('Preprocessor', self.preprocessor),
                ('Fix Names', FunctionTransformer(clean_column_names)),
                ('Add Features', FunctionTransformer(feature_adder)),
            ])
        if smote:
            predict_pipeline.steps.append(('SMOTE', SMOTE(sampling_strategy=smote_sampling_strategy, k_neighbors=5)))
        if under_sample:
            predict_pipeline.steps.append(('Under Sample', RandomUnderSampler()))

        if model_name == 'XGBoost':
            clf = XGBClassifier(**params)
        elif model_name == 'CatBoost':
            clf = CatBoostClassifier(**params)
        elif model_name == 'Random Forest':
            clf = RandomForestClassifier(**params)
        predict_pipeline.steps.append((model_name, clf))

        confusion_matrices = []
        
        scores = []

        for train_index, test_index in self.cv.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]            
            
            if class_weight:
                class_weights = compute_sample_weight(
                class_weight = class_weight,
                y=y_train
            )
                predict_pipeline.fit(X_train, y_train, **{f'{model_name}__sample_weight': class_weights})
            else:
                predict_pipeline.fit(X_train, y_train)

            y_pred_proba = predict_pipeline.predict_proba(X_test)
            y_pred = predict_pipeline.predict(X_test)
            scores.append(roc_auc_score(y_test, y_pred_proba, multi_class='ovr'))
            

            cm = confusion_matrix(y_test, y_pred)
            confusion_matrices.append(cm)

        mean_score = np.mean(scores)
        print(mean_score)
        sum_confusion_matrix = np.sum(confusion_matrices, axis=0)

        row_sums = sum_confusion_matrix.sum(axis=1, keepdims=True)
        
        average_confusion_matrix = (sum_confusion_matrix / row_sums)

        if display:
            cm = plt.imshow(average_confusion_matrix, cmap=plt.cm.Blues, interpolation='nearest')
            plt.colorbar()
            class_names = ['A', 'C', 'D', 'L', 'Q','S', 'V', 'X']

            for i in range(average_confusion_matrix.shape[0]):
                for j in range(average_confusion_matrix.shape[1]):
                    plt.text(j, i, f"{average_confusion_matrix[i, j]:.2f}",
                        ha="center", va="center", color="black", fontsize=10)
                    
            ax = plt.gca()
            ax.set_xticks(np.arange(len(class_names)))
            ax.set_yticks(np.arange(len(class_names)))
            ax.set_xticklabels(class_names)
            ax.set_yticklabels(class_names)

            plt.title(f'Averaged Confusion Matrix ({model_name})')
            plt.xlabel("Predicted label")
            plt.ylabel("True label")
            plt.tight_layout()
            plt.show()

        return average_confusion_matrix, np.mean(scores)

    def tune(self, model_name, display):
        study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))

        smote='auto' # dict or 'auto'
        class_weight=None # list or 'balanced'

        study.optimize(lambda trial: self.objective(trial, model_name, smote, class_weight), n_trials=self.n_trials)

        with open(f'project/study_dumps/{model_name}_study.pkl', 'wb') as f:
            dill.dump(study, f)

        best_params = study.best_params
        best_score = study.best_value

        print(f'Best Score: {best_score}\n\
              Best Params: {best_params}')
        
        logging.info(f'Score: {best_score}\nParams:{best_params}\n')
        
        if display:
            fig1 = plot_optimization_history(study)
            plt.show()
            fig2 = plot_param_importances(study)
            plt.show()
        
        return best_params, best_score, study
    
transformer = DataTransformer(
    remove_cols=['pdes', 'class', 'class2', 'score', 'bad', 'neo', 'rot_per', 'mean_anomaly', 'abs_mag', 'peri', 'asc_node_long'],
    target_name = 'class1', 
    num_imputer='iterative', 
    scaling='standard', 
    cat_encoding='ohe',
    add_features=True,
    excluded_features = []
)

preprocessor = transformer.establish_pipeline()

MT = ModelTuner(130, preprocessor, StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED))

#params, score, study = MT.tune('XGBoost', display=True)

sample_strat = { #this is for smote
        0: 2591,
        1: get_sample_ratio('C', .82),
        2: 3387,
        3: 4966,
        4: 1787,
        5: get_sample_ratio('S', .82),
        6: 1726,
        7: 3268
        }

best_class_weights = { #this is for class weighting
    0: 6.151603987284707, 
    1: 8.363613395213525, 
    2: 2.176918429551468, 
    3: 9.187557677437756, 
    4: 0.35935486278362616, 
    5: 5.236849237930324, 
    6: 6.382881361581345, 
    7: 2.099174334245832
}


#MT.cv_matrix('Random Forest', class_weight=None, params = {}, 
#                  smote=True, smote_sampling_strategy='auto', under_sample=False, display=True)

best_xgb_params = {
    'objective': 'multi:softprob',
    'random_state': RANDOM_SEED,
    'learning_rate': 0.099, 
    'max_depth': 5,
    'min_child_weight': 0.2784016726023859,
    'subsample': 0.4, 
    'gamma': 1.8685907147813174
    }

best_catboost_params = {'learning_rate': 0.01, 'n_estimators': 512, 'max_depth': 8, 'l2_leaf_reg': 0.5433106820990699, 'rsm': 0.6973423627892332}
best_randomforest_params = {'n_estimators': 462, 'max_depth': 10, 'min_samples_leaf': 6, 'min_samples_split': 15} # SMOTE K=24
MT.tune('Random Forest', True) # trained with balanced smote




