import numpy as np
import pandas as pd

from sklearn import set_config

import os
import sys

from src.config import *
from src.utils import *

from src.exception import CustomException

class PredictPipeline:
    """
    A class for making predictions using an ensemble model.
    """
    def __init__(self):
        set_config(transform_output="pandas")

    def predict(self, features, decode_label=True, predict_proba=False):
        """
        Predicts using the ensemble model.

        :param features: The input features for prediction.
        :param decode_label: Whether to decode the predicted labels.
        :param predict_proba: Whether to return probabilities.
        :return: Predicted labels or probabilities.
        """
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
    """
    A class for representing custom data for prediction.
    """
    def __init__(self,
        osc_semimaj_ax:float, 
        osc_inclin:float,
        diameter:float, 
        albedo:float,
        osc_eccentricity: float
    ):
        """
        Initialize the CustomData object.

        :param osc_semimaj_ax: The osculating semimajor axis.
        :param osc_inclin: The osculating inclination.
        :param diameter: The diameter of the object.
        :param albedo: The albedo of the object.
        :param osc_eccentricity: The osculating eccentricity.
        """
        self.osc_semimaj_ax = osc_semimaj_ax
        self.osc_inclin = osc_inclin
        self.diameter = diameter
        self.albedo = albedo
        self.osc_eccentricity = osc_eccentricity
    
    def get_data_as_df(self):
        """
        Get the custom data as a DataFrame.

        :return: The data as a DataFrame.
        """
        data_dict = {
            'osc_semimaj_ax':[self.osc_semimaj_ax],
            'osc_inclin':[self.osc_inclin],
            'diameter':[self.diameter],
            'albedo':[self.albedo],
            'osc_eccentricity':[self.osc_eccentricity]
        }

        return pd.DataFrame(data_dict)

