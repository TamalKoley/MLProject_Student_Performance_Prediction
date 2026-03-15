import os;
import sys;
import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
import seaborn as sns;
from dataclasses import dataclass;
from catboost import CatBoostRegressor;
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor;
from sklearn.linear_model import LinearRegression;
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error;
from sklearn.neighbors import KNeighborsRegressor;
from sklearn.tree import DecisionTreeRegressor;
from xgboost import XGBRegressor
from typing import Tuple,Dict,List;

from src.exception import CustomException;
from src.logger import logging;
from src.utils import save_object;

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','trained_model.pkl')
    model_score_file=os.path.join('artifacts','model_score.txt')
class ModelTrainer:
    def __init__(self):
        ####Constructor to intialize basic configs and models and hyperparameters
        self.__model_trainer_config=ModelTrainerConfig()
        self.__models={
                'CatBoost' :CatBoostRegressor(),
                'Adaboost' : AdaBoostRegressor(),
                'GradientBoost' : GradientBoostingRegressor(),
                'Random_Forest' : RandomForestRegressor(),
                'Linear_Regression' : LinearRegression(),
                'K_Neighbour' :KNeighborsRegressor(),
                'Decision_Tree' : DecisionTreeRegressor(),
                'XGBoost' : XGBRegressor()
            };

    def __model_scoring(self,y_train:np.ndarray,y_train_pred:np.ndarray,y_test:np.ndarray,y_test_pred:np.ndarray)->Tuple[Tuple[float,float,float],Tuple[float,float,float]]:
        ##### This fuction will calculate model performance based on predictions
        try:
            train_mse=mean_squared_error(y_train,y_train_pred)
            train_mae=mean_absolute_error(y_train,y_train_pred)
            train_score=r2_score(y_train,y_train_pred)
            test_mse=mean_squared_error(y_test,y_test_pred)
            test_mae=mean_absolute_error(y_test,y_test_pred)
            test_score=r2_score(y_test,y_test_pred)
            training_result=(train_mse,train_mae,train_score)
            test_result=(test_mse,test_mae,test_score)
            return(
                    training_result,
                    test_result
            );

        except Exception as e:
            raise CustomException(e,sys)
        

    def __model_training(self,models:dict,x_train:np.ndarray,y_train:np.ndarray,x_test:np.ndarray,y_test:np.ndarray)->Dict[str,float]:
        ###### this function will train all the models based on train and test data and will make prediction
        try:
            model_scores={};
            for key in models.keys():
                logging.info(f'Model is training for {key} ')
                model=models[key];
                model.fit(x_train,y_train)
                y_train_pred=model.predict(x_train)
                y_test_pred=model.predict(x_test)
                logging.info(f'Model training completed for {key} ')
                logging.info(f'Calculating Model performance for {key} ')
                training_result,test_result=self.__model_scoring(y_train=y_train,y_train_pred=y_train_pred,y_test=y_test,y_test_pred=y_test_pred)
                logging.info(f'Model performance calculation completed for {key} ')
                train_mse,train_mae,train_score=training_result
                test_mse,test_mae,test_score=test_result
                writebuffer=f"""***********Model :{key}**********\n----------- Model Performance on Train data ---------\nMSE : {train_mse}\nMAE : {train_mae}\nScore : {train_score}\n----------- Model Performance on Test data ---------\nMSE : {test_mse}\nMAE : {test_mae}\nScore : {test_score}\n"""
                save_object(writebuffer,self.__model_trainer_config.model_score_file,format='text',mode='append')
                logging.info(f'Model performance data saved for {key} ')
                model_scores[key]=test_score;
            return model_scores;
                
        except Exception as e:
            raise CustomException(e,sys)



    def initiate_model_training(self,x_train,y_train,x_test,y_test)->List[str]:
         ###### this function will start the model training by initializing models and calling model_training fucntion
        try:
            logging.info('Model Training Started')

            model_scores=self.__model_training(models=self.__models,x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test)
            logging.info('Model Training completed')
            sorted_model_scores=dict(sorted(model_scores.items() ,key=lambda item : item[1], reverse=True))
            best_models=[];
            #print('Best Model perfomanance values');
            for i,(key,value) in enumerate(sorted_model_scores.items()):
                if i<3:
                    best_models.append(key);
            return best_models;
        except Exception as e:
            raise CustomException(e,sys)