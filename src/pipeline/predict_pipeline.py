import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("output","model.pkl")
            preprocessor_path=os.path.join('output','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(  self,
        mean_radius: float,
        mean_texture: float,
        mean_perimeter: float,
        mean_area: float,
        mean_smoothness: float,
        mean_compactness: float,
        mean_concavity: float, 
        mean_concave_points: float,
        mean_symmetry: float,
        mean_fractal_dimension: float,
        radius_error: float,
        texture_error: float,
        perimeter_error: float,
        area_error: float,
        smoothness_error: float,
        compactness_error: float,
        concavity_error: float,
        concave_points_error: float,
        symmetry_error: float,
        fractal_dimension_error: float,
        worst_radius: float,
        worst_texture: float,
        worst_perimeter: float,
        worst_area: float,
        worst_smoothness: float,
        worst_compactness: float,
        worst_concavity: float,
        worst_concave_points: float,
        worst_symmetry: float,
        worst_fractal_dimension: float):

        self.mean_radius = mean_radius

        self.mean_texture = mean_texture

        self.mean_perimeter = mean_perimeter

        self.mean_area = mean_area

        self.mean_smoothness = mean_smoothness

        self.mean_compactness = mean_compactness

        self.mean_concavity = mean_concavity

        self.mean_concave_points = mean_concave_points

        self.mean_symmetry = mean_symmetry

        self.mean_fractal_dimension = mean_fractal_dimension

        self.radius_error = radius_error

        self.texture_error = texture_error

        self.perimeter_error = perimeter_error

        self.area_error = area_error

        self.smoothness_error = smoothness_error

        self.compactness_error = compactness_error

        self.concavity_error = concavity_error

        self.concave_points_error = concave_points_error

        self.symmetry_error = symmetry_error

        self.fractal_dimension_error = fractal_dimension_error

        self.worst_radius = worst_radius

        self.worst_texture = worst_texture

        self.worst_perimeter = worst_perimeter
        
        self.worst_area = worst_area

        self.worst_smoothness = worst_smoothness

        self.worst_compactness = worst_compactness

        self.worst_concavity = worst_concavity

        self.worst_concave_points = worst_concave_points

        self.worst_symmetry = worst_symmetry

        self.worst_fractal_dimension = worst_fractal_dimension

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "mean_radius": [self.mean_radius],
                "mean_texture": [self.mean_texture],
                "mean_perimeter": [self.mean_perimeter],
                "mean_area": [self.mean_area],
                "mean_smoothness": [self.mean_smoothness],
                "mean_compactness": [self.mean_compactness],
                "mean_concavity": [self.mean_concavity],
                "mean_concave_points": [self.mean_concave_points],
                "mean_symmetry": [self.mean_symmetry],
                "mean_fractal_dimension": [self.mean_fractal_dimension],
                "radius_error": [self.radius_error],
                "texture_error": [self.texture_error],
                "perimeter_error": [self.perimeter_error],
                "area_error": [self.area_error],
                "smoothness_error": [self.smoothness_error],
                "compactness_error": [self.compactness_error],
                "concavity_error": [self.concavity_error],
                "concave_points_error": [self.concave_points_error],
                "symmetry_error": [self.symmetry_error],
                "fractal_dimension_error": [self.fractal_dimension_error],
                "worst_radius": [self.worst_radius],
                "worst_texture": [self.worst_texture],
                "worst_perimeter": [self.worst_perimeter],
                "worst_area": [self.worst_area],
                "worst_smoothness": [self.worst_smoothness],
                "worst_compactness": [self.worst_compactness],
                "worst_concavity": [self.worst_concavity],
                "worst_concave_points": [self.worst_concave_points],
                "worst_symmetry": [self.worst_symmetry],
                "worst_fractal_dimension": [self.worst_fractal_dimension]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)