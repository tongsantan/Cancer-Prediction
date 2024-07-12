## 5. Update the components
import os
import sys
from dataclasses import dataclass
from src.entity.config_entity import DataTransformationConfig
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.exception import CustomException
from src import logger
from src.utils.common import save_object

## 5. Update the components

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config=config

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        
        '''
        try:
            numerical_columns = ['mean_radius', 'mean_texture', 'mean_smoothness', 'mean_compactness',
       'mean_concavity', 'mean_concave_points', 'mean_symmetry']
            
            num_pipeline= Pipeline(
                    steps=[
                            ("scaler",StandardScaler())
                    ]
                )

            logger.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                    [
                        ("num_pipeline",num_pipeline,numerical_columns),
                    ]
                )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self):

        try:
            train_df=pd.read_csv(self.config.train_data_path)
            test_df=pd.read_csv(self.config.test_data_path)

            logger.info("Read train and test data completed")

            logger.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="malignant"

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logger.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            np.save(self.config.train_array_path, train_arr)
            np.save(self.config.test_array_path, test_arr)


            logger.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                self.config.train_array_path,
                self.config.test_array_path,
                self.config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)