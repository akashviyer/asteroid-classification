import numpy as np
import pandas as pd

from sklearn import set_config

import os
import sys

from src.config import *
from src.utils import *

from src.exception import CustomException

class PredictPipeline:
    def __init__(self):
        set_config(transform_output="pandas")

    def predict(self, features, decode_label=True, predict_proba=False):
        try:
            model_path = os.path.join('project', 'artifacts', 'ensemble.pkl')
            model = load_object(file_path=model_path)
            if not predict_proba:
                preds = model.predict(features)
                if decode_label:
                    preds = get_str_label(preds)
            else:
                preds = model.predict_proba(features)
            
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
        osc_semimaj_ax:float, 
        osc_inclin:float,
        diameter:float, 
        albedo:float,
        osc_eccentricity: float
    ):
        self.osc_semimaj_ax = osc_semimaj_ax
        self.osc_inclin = osc_inclin
        self.diameter = diameter
        self.albedo = albedo
        self.osc_eccentricity = osc_eccentricity
    
    def get_data_as_df(self):
        data_dict = {
            'osc_semimaj_ax':[self.osc_semimaj_ax],
            'osc_inclin':[self.osc_inclin],
            'diameter':[self.diameter],
            'albedo':[self.albedo],
            'osc_eccentricity':[self.osc_eccentricity]
        }

        return pd.DataFrame(data_dict)
