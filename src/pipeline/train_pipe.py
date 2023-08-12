import numpy as np
import pandas as pd

import os
import sys

from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
)

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn import set_config
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from sklearn.utils.class_weight import compute_sample_weight

from src.utils import *
from src.config import *

from src.exception import CustomException

set_config(transform_output="pandas")

class ModelEnsemble:
    def __init__(self, fitted_models, model_weights):
        self.fitted_models = fitted_models
        self.model_weights = model_weights

    def predict_proba(self, data):
        model_names = list(self.fitted_models.keys())
        
        agg_probs = None

        for model_name in model_names:
            model = self.fitted_models[model_name]
            weight = self.model_weights[model_name]
            
            weighted_probs = model.predict_proba(data) * weight

            if agg_probs is None:
                agg_probs = weighted_probs
            else:
                agg_probs += weighted_probs
                
        return agg_probs
            
    def predict(self, data):
        '''
        Predicts target variable for input data,
        using ensemble based on specifications
        '''
        predict_proba_output = self.predict_proba(data)

        preds = np.argmax(predict_proba_output, axis=1)

        return preds

class ModelEnsembler:
    def __init__(self, model_names:list, model_configs:dict):
        self.model_names = model_names
        self.model_configs = model_configs
        self.preprocessor_filepath = os.path.join('project', 'artifacts', 'preprocessor.pkl')
        self.preprocessor = load_object(self.preprocessor_filepath)
        self.ensemble_filepath = os.path.join('project', 'artifacts', 'ensemble.pkl')

    def assemble_pipelines(self):
        pipeline_dict = {}

        for model_name in self.model_names:
            configs = self.model_configs[model_name]

            smote = configs['smote']
            smote_sample_counts = configs['smote_sampling_strategy']['sample_counts']
            smote_k = configs['smote_sampling_strategy']['k']

            under_sample = configs['under_sample']
            params = configs['params']

            predict_pipeline = Pipeline(steps=[
                    ('Preprocessor', self.preprocessor),
                    ('Fix Names', FunctionTransformer(clean_column_names)),
                    ('Add Features', FunctionTransformer(feature_adder)),
                ])
            
            if smote:
                predict_pipeline.steps.append(('SMOTE', SMOTE(sampling_strategy=smote_sample_counts, k_neighbors=smote_k)))

            if under_sample:
                predict_pipeline.steps.append(('Under Sample', RandomUnderSampler()))

            if model_name == 'XGBoost':
                clf = XGBClassifier(**params)
            elif model_name == 'CatBoost':
                clf = CatBoostClassifier(**params)
            elif model_name == 'RandomForest':
                clf = RandomForestClassifier(**params)

            predict_pipeline.steps.append((model_name, clf))

            pipeline_dict[model_name] = predict_pipeline

        return pipeline_dict
    
    def train_pipelines(self):
        train_df = get_train_df()
        X, y = get_X_and_y(train_df, TARGET)
        y = label_encode(y)

        pipeline_dict = self.assemble_pipelines()

        fitted_models = {}

        for model_name in self.model_names:
            model = pipeline_dict[model_name]
            class_weight = self.model_configs[model_name]['class_weight']

            if class_weight:
                class_weights = compute_sample_weight(
                class_weight = class_weight,
                y=y
                )

                model.fit(X, y, **{f'{model_name}__sample_weight': class_weights})
            else:
                model.fit(X, y)

            fitted_models[model_name] = model

        return fitted_models
        
    def save_ensemble(self):
        fitted_models = self.train_pipelines()
        model_weights = self.model_configs['model_weights']

        ensemble = ModelEnsemble(fitted_models, model_weights)

        save_object(self.ensemble_filepath, ensemble)

        return self.ensemble_filepath

#ModelEnsembler(IMPLEMENTED_MODELS, model_configs=BEST_MODEL_CONFIGS).save_ensemble()

