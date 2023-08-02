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
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, FunctionTransformer
from scikeras.wrappers import KerasClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging

from data_transformation import DataTransformer

from src.utils import *
from src.config import *

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

@dataclass
class ModelDeployerConfig:
    train_df = get_train_df()

    X, y = get_X_and_y(train_df, TARGET)

    transformer = transformer
    preprocessor = preprocessor
    models = {
        #'CatBoost' : CatBoostClassifier(verbose=False, random_state=RANDOM_SEED),
        #'Decision Tree' : DecisionTreeClassifier(random_state=RANDOM_SEED),
        'XGBoost' : XGBClassifier(objective='multi:softprob', tree_method='hist', enable_categorical=True, random_state=RANDOM_SEED),
        #'Adaboost' : AdaBoostClassifier(n_estimators=75, learning_rate=0.75, random_state=RANDOM_SEED),
        #'Random Forest' : RandomForestClassifier(random_state=RANDOM_SEED),
        #'KNN': KNeighborsClassifier(),
        #'ANN' : KerasClassifier(build_fn=specific_nn_generator(len(transformer.feature_list), len(np.unique(y))), epochs=5, batch_size=32)
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True)


class ModelDeployer:
    def __init__(self):
        self.config = ModelDeployerConfig()

    def cv_models(self):
        model_cv_scores = {}

        predict_pipeline = Pipeline(steps=[
                ('Preprocessor', self.config.preprocessor),
                ('Fix Names', FunctionTransformer(clean_column_names))
            ])
        
        if self.config.transformer.add_features:
            predict_pipeline.steps.append(('Add Features', FunctionTransformer(feature_adder)))
            #predict_pipeline.steps.append(('Add Polynomial Features', FunctionTransformer(polynomial_feature_adder)))

        if self.config.transformer.excluded_features:
            predict_pipeline.steps.append(('Exclude Features', FunctionTransformer(feature_excluder(self.config.transformer.excluded_features))))

        for model_name, model in self.config.models.items():
            predict_pipeline.steps.append((model_name, model))
            if model_name == 'XGBoost' or model_name == 'CatBoost':
                label_encoder = LabelEncoder().fit(self.config.y)
                y_encoded = label_encoder.transform(self.config.y)
                cv_results = cross_validate(predict_pipeline, self.config.X, y_encoded, cv=self.config.cv, scoring='roc_auc_ovr')
            elif model_name == 'ANN':
                cv_results = cross_validate(predict_pipeline, self.config.X, self.config.y, cv=self.config.cv, scoring='accuracy')
            else:
                cv_results = cross_validate(predict_pipeline, self.config.X, self.config.y, cv=self.config.cv, scoring='roc_auc_ovr')
            model_cv_scores[model_name] = cv_results['test_score']

            predict_pipeline.steps.pop(-1)
        
        return model_cv_scores

    def plot_model_mean_performance(self):
        model_scores = self.cv_models()
        model_names = model_scores.keys()
        cv_scores = model_scores.values()
        avg_scores = [np.mean(scores) for scores in cv_scores]

        plt.figure(figsize=(20, 15))
        model_hist = sns.barplot(x=list(model_names), y=avg_scores)

        model_hist.set_xticklabels(model_hist.get_xticklabels(), rotation=45)

        model_cv_details_string = '\n' + str(self.config.transformer) + '\n' + get_model_scores_and_params_string(self.config, model_names, avg_scores)

        logging.info(model_cv_details_string)

        plt.show()
    
    def plot_feature_importance(self, model_name, display=True, ax=None, add_noise=False):
        model = self.config.models[model_name]
        predict_pipeline = Pipeline(steps=[
            ('Preprocessor', self.config.preprocessor),
            ('Fix Names', FunctionTransformer(clean_column_names))
            ])
        
        if self.config.transformer.add_features:
            predict_pipeline.steps.append(('Add Features', FunctionTransformer(feature_adder)))
            #predict_pipeline.steps.append(('Add Polynomial Features', FunctionTransformer(polynomial_feature_adder)))

        if add_noise:
            predict_pipeline.steps.append(('Add Noise', FunctionTransformer(add_noise_feature)))

        predict_pipeline.steps.append((model_name, model))

        y = label_encode(self.config.y)

        predict_pipeline.fit(self.config.X, y)
        
        if model_name == 'XGBoost':
            features = model.get_booster().feature_names
            importances = model.feature_importances_
        elif model_name == 'CatBoost':
            features = model.feature_names_
            importances = model.get_feature_importance()
        else:
            features = model.feature_names_in_
            importances = model.feature_importances_
        
        sorted_idx = importances.argsort()

        features = np.array(features)
        importances = np.array(importances)

        fi = sns.barplot(x=features[sorted_idx], y=importances[sorted_idx], ax=ax)
        fi.set_xticklabels(fi.get_xticklabels(), rotation=90, fontsize=6)
        fi.bar_label(fi.containers[0], rotation=270, fontsize=7)


        if display: 
            plt.show()

        return fi
    
    def plot_all_model_importances(self, add_noise=False, save_figure=False):
        model_names = list(self.config.models.keys())
        try:
            model_names.remove('ANN')
            model_names.remove('KNN')
        except:
            pass
        fig, axs = plt.subplots(1, len(list(model_names)), figsize=(20, 15))
        for i, model_name in enumerate(model_names):
            fi_plot = self.plot_feature_importance(model_name, display=False, ax=axs[i], add_noise=add_noise)
            fi_plot.set_title(model_name)
        plt.tight_layout()
        plt.show()

        if save_figure:
            plt.savefig(f'project/visuals/importances_{hash(fi_plot)}.png')
        return

model_deployer = ModelDeployer()

model_deployer.plot_model_mean_performance()

#model_deployer.plot_all_model_importances(add_noise=True, save_figure=False)