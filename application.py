from flask import Flask,request,render_template;
import sys;
from src.exception import CustomException;
from src.pipeline.predict_pipeline import PredictionPipeLine;
from src.logger import logging;
import pandas as pd;

application=Flask(__name__);
app=application;


@app.route('/',methods=['GET','POST'])
def prediction():
    logging.info('app started on port 10222')
    try:
        if request.method=='GET':
            return render_template('index.html')
        else:
            gender=request.form.get('gender')
            race_ethnicity=request.form.get('race_ethnicity')
            parental_level_of_education=request.form.get('parental_level_of_education')
            lunch=request.form.get('lunch')
            test_preparation_course=request.form.get('test_preparation_course')
            reading_score=request.form.get('reading_score')
            writing_score=request.form.get('writing_score')
            prediction_obj=PredictionPipeLine()
            logging.info('app is predicting the result')
            result=prediction_obj.model_prediction(
                pd.DataFrame({
                'gender':[gender],
                'race_ethnicity':[race_ethnicity],
                'parental_level_of_education':[parental_level_of_education],
                'lunch':[lunch],
                'test_preparation_course':[test_preparation_course],
                'reading_score':[reading_score],
                'writing_score':[writing_score],
                })
                )
            logging.info('prediction is completed')
            return render_template('index.html',results=result)
    except CustomException as e:
        logging.info(e.error_message)



if __name__=='__main__':
    app.run(host='0.0.0.0',port=10222)