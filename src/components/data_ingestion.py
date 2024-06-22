import os
import sys
from src.exception import CustomException
from src import logger
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedShuffleSplit
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    processed_data_path: str=os.path.join('data', "processed_cancer.csv")
    train_data_path: str=os.path.join('output',"train.csv")
    test_data_path: str=os.path.join('output',"test.csv")
    raw_data_path: str=os.path.join('output',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logger.info("Initiate data ingestion method or component")
        try:
            df_bc = load_breast_cancer() 
            
            data = np.c_[df_bc.data, df_bc.target]
            
            column_names = np.append(df_bc.feature_names, ['malignant'])
            
            df = pd.DataFrame(data, columns=column_names)

            df = df.drop(columns = ['mean perimeter', 'mean area', 'worst radius', 'worst perimeter', 'worst area', 'fractal dimension error', 'mean fractal dimension','radius error',
                        'texture error', 'smoothness error', 'symmetry error', 'worst texture', 'worst smoothness', 'worst compactness', 'worst concavity', 'worst concave points', 
                        'worst symmetry', 'worst fractal dimension', 'perimeter error', 'area error', 'concavity error', 'concave points error', 'compactness error'])
            
            df.columns = df.columns.str.replace(' ', '_')
            
            df['malignant'] = df['malignant'].map(lambda x: 1 if x != 1.0 else 0)
            
            os.makedirs(os.path.dirname(self.ingestion_config.processed_data_path),exist_ok=True)
            
            df.to_csv(self.ingestion_config.processed_data_path,index=False,header=True)

            return(
                self.ingestion_config.processed_data_path
            )
        
        except Exception as e:
            raise CustomException(e,sys) 

    def complete_data_ingestion(self):
        logger.info("Resume data ingestion method or component") 

        try:
            df=pd.read_csv('./data/processed_cancer.csv')
            
            logger.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            strat_shuff_split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        
            X = df.drop(columns=['malignant'],axis=1)
            
            y = df['malignant']
            
            train_idx, test_idx = next(strat_shuff_split.split(X, y))
            
            train_set = df.loc[train_idx]
            
            test_set = df.loc[test_idx]

            logger.info("Train test split initiated")

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logger.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)



