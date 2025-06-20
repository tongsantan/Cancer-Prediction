import sys
import pandas as pd
from src.exception import CustomException
from src.utils.common import load_object
import os

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("output", "model_trainer", "model.pkl")
            preprocessor_path=os.path.join('output', 'data_transformation', 'preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)


['mean_radius', 'mean_texture', 'mean_smoothness', 'mean_compactness',
       'mean_concavity', 'mean_concave_points', 'mean_symmetry']

class CustomData:
    def __init__(  self,
        mean_radius: float,
        mean_texture: float,
        mean_smoothness: float,
        mean_compactness: float,
        mean_concavity: float, 
        mean_concave_points: float,
        mean_symmetry: float):

        self.mean_radius = mean_radius

        self.mean_texture = mean_texture

        self.mean_smoothness = mean_smoothness

        self.mean_compactness = mean_compactness

        self.mean_concavity = mean_concavity

        self.mean_concave_points = mean_concave_points

        self.mean_symmetry = mean_symmetry

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "mean_radius": [self.mean_radius],
                "mean_texture": [self.mean_texture],
                "mean_smoothness": [self.mean_smoothness],
                "mean_compactness": [self.mean_compactness],
                "mean_concavity": [self.mean_concavity],
                "mean_concave_points": [self.mean_concave_points],
                "mean_symmetry": [self.mean_symmetry]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)