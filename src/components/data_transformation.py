import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn import set_config
from sklearn.compose import ColumnTransformer
#from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import *
from src.config import *

# Set output type for all transformers to pandas dataframe
set_config(transform_output="pandas")

@dataclass
class DataTransformerConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformer:
    def __init__(self, add_features=True, excluded_features=[], target_name=TARGET, remove_cols=[], num_imputer='None', scaling='None', cat_encoding='None'):
        self.target_name = target_name
        self.remove_cols = remove_cols
        self.feature_list = get_train_df().drop(target_name, axis=1).drop(remove_cols, axis=1).columns.tolist()
        self.num_imputer = num_imputer
        self.scaling = scaling
        self.cat_encoding = cat_encoding
        self.add_features = add_features
        self.excluded_features = excluded_features
        self.config = DataTransformerConfig()

    def __str__(self):return f"\
            Removed columns: {self.remove_cols}\n\
            Numerical Imputer: {self.num_imputer}\n\
            Numerical Scaler: {self.scaling}\n\
            Categorical_Encoder: {self.cat_encoding}\
            "

    def establish_pipeline(self):
        '''
        Establishes a data transformation pipeline using the columns of the train data csv,
        without the specified columns from [remove_cols]
        '''
        try:
            # Lists for names of numerical and categorical columns in X
            categorical_cols = ['neo', 'pdes', 'class', 'class2']
            categorical_cols = [col for col in categorical_cols if col not in self.remove_cols]

            numerical_cols = [col for col in self.feature_list if col not in categorical_cols]
            [col for col in numerical_cols if col not in self.remove_cols]

            num_pipe = Pipeline(
                steps=[
                ]
            )

            if self.num_imputer == 'mean':
                num_pipe.steps.append(('Mean Imputer', SimpleImputer()))

            elif self.num_imputer == 'iterative':
                num_pipe.steps.append(('Iterative Imputer', IterativeImputer()))

            if self.scaling == 'standard':
                num_pipe.steps.append(('Standard Scaler', StandardScaler()))
            elif self.scaling == 'minmax':
                num_pipe.steps.append(('MinMax Scaler', MinMaxScaler()))

            cat_pipe = Pipeline(
                steps=[]
            )

            if self.cat_encoding == 'ohe':
                cat_pipe.steps.append(('One Hot Encoder', OneHotEncoder(drop='if_binary', sparse_output=False)))


            preprocessor = ColumnTransformer(
                transformers=[
                    ('Numerical', num_pipe, numerical_cols),
                    ('Categorical', cat_pipe, categorical_cols),
                ]
            )

            logging.info('Data transformation pipeline created successfully')

            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def save_transform(self):
        try:
            train = get_train_df()
            test = get_test_df()

            if self.remove_cols:
                    train = train.drop(self.remove_cols, axis=1)
                    test = test.drop(self.remove_cols, axis=1)

            train_X, train_y = train.drop(self.target_name, axis=1), train[self.target_name]

            test_X, test_y = test.drop(self.target_name, axis=1), test[self.target_name]

            logging.info('Train and test data read-in as dataframes, and train split into X and y')

            logging.info('Initializing preprocessor')

            preprocessor = self.establish_pipeline()

            logging.info('Preprocessor initialized')

            transformed_X_train = preprocessor.fit_transform(train_X, train_y)
            transformed_X_test = preprocessor.transform(test_X)

            train_arr = np.c_[
                transformed_X_train, np.array(train_y)
            ]
            test_arr = np.c_[
                transformed_X_test, np.array(test_y)
                ]

            save_object(
                    file_path=self.config.preprocessor_obj_file_path,
                    obj=preprocessor
                )
            
            logging.info('Preprocessor object saved')

            return (
                train_arr,
                test_arr,
                self.config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)

data_transformer = DataTransformer(
    target_name = 'class1',
    remove_cols = ['pdes', 'class', 'class2', 'score', 'bad', 'neo'],
    num_imputer = 'mean',
    scaling = 'standard',
    cat_encoding = 'ohe'
)

#train_input, test_input, filepath = data_transformer.save_transform()