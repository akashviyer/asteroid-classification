import os
import sys

from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    logging.info('Data Ingestion Started')
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def clean_data(self, df):
        df = df.drop(['prop_eccen', 'prop_inclin_sin', 'prop_semimaj_ax', 'epoch_mjd'], axis=1)
        df['osc_eccentricity'] = (df['osc_eccen'] + df['eccentricity']) / 2
        df = df.drop(['osc_eccen', 'eccentricity'], axis=1)

        df['osc_semimaj_ax'] = (df['osc_semimaj_ax'] + df['semimaj_ax2']) / 2
        df = df.drop('semimaj_ax2', axis=1)

        df['osc_inclin'] = (df['osc_inclin'] + df['inclin']) / 2
        df = df.drop('inclin', axis=1)

        df['class1'] = df['class'].str[0]
        df['class2'] = df['class'].str[-1]

        return df

    def initiate_data_ingestion(self):
        try:
            df=pd.read_csv('C:/Users/rules/OneDrive/Desktop/Projects/project/notebook/data/raw_data.csv')
            logging.info('Read dataset into dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df = self.clean_data(df)
            logging.info('Cleaned data')

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train, test = train_test_split(df, test_size=0.2, stratify=df['class1'], random_state=42)

            train.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            test.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Data ingestion complete')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__=="__main__":
    obj=DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()