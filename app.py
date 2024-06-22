from flask import Flask,request,render_template
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application



## Route for a home page

@app.route('/')
def index():
    return render_template('home.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            mean_radius=float(request.form.get('mean_radius')),
            mean_texture=float(request.form.get('mean_texture')),
            mean_smoothness=float(request.form.get('mean_smoothness')),
            mean_compactness=float(request.form.get('mean_compactness')),
            mean_concavity=float(request.form.get('mean_concavity')),
            mean_concave_points=float(request.form.get('mean_concave_points')),
            mean_symmetry=float(request.form.get('mean_symmetry'))
        )

        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0") 

   