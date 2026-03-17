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
from sklearn.model_selection import RandomizedSearchCV;

from src.exception import CustomException;
from src.logger import logging;
from src.utils import save_object;

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','trained_model.pkl')
    model_score_file=os.path.join('artifacts','model_score.txt')
    hyper_parameter_result_file=os.path.join('artifacts','hyperparameter_result.txt')
    final_model_file_path=os.path.join('artifacts','final_model.pkl')
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
        self.__hyper_params={
                'CatBoost' :{
                        'iterations': [100, 200, 250,300],
                        'learning_rate': [0.01, 0.05, 0.1,1.0],
                        'depth': [4, 6, 10,13,18,25,35],
                        'l2_leaf_reg': [1, 3, 5, 9]
                },
                'Adaboost' : {
                    'n_estimators' : [25,50,100,150,200],
                    'learning_rate' : [1.0,0.1,0.01,0.001],
                    'loss' : ['linear','square','exponential']

                },
                'GradientBoost' : {
                        'loss' : ['squared_error', 'absolute_error', 'huber', 'quantile'],
                        'learning_rate' : [1.0,0.1,0.01,0.001],
                        'n_estimators' : [25,50,100,150,200],
                        'criterion' : ['friedman_mse','squared_error'],
                        'max_depth' :[7,15,20,30,35]

                },
                'Random_Forest' : {
                    'n_estimators' : [25,50,100,150,200],
                    'criterion' : ['squared_error','absolute_error','friedman_mse','poisson'],
                    'max_depth' :[7,15,20,30,35],
                    'min_samples_split' : [2,5,10,17,20]

                },
                'Linear_Regression' : {},
                'K_Neighbour' :{},
                'Decision_Tree' : {},
                'XGBoost' : {}
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
    

    def initiate_hyperparameter_tunning(self,best_models:List,x_train:np.ndarray,y_train:np.ndarray)->List[Dict]:
        ##### This function will perform hyper paramete tunning of the selected models.
        try:
            logging.info('Hyper Parameter Tunning Started')
            tunning_results=[];
            for model in best_models:
                logging.info(f'Hyper Parameter Tunning Started for {model}')
                hypermodel=self.__models[model]
                model_params=self.__hyper_params[model];
                random_cv=RandomizedSearchCV(
                    estimator=hypermodel,
                    param_distributions=model_params,
                    n_iter=100,
                    cv=3,
                    n_jobs=-1,
                    verbose=2
                );
                random_cv.fit(x_train,y_train)
                tunning_results.append({
                    'model_name' : model,
                    'best_score' : random_cv.best_score_,
                    'best_params' : random_cv.best_params_
                })
            logging.info('Hyper Parameter Tunning Completed')
            logging.info('Saving Hyper Parameter Tunning result')
            save_object(object=tunning_results,filename=self.__model_trainer_config.hyper_parameter_result_file,format='text',mode='write')
            return tunning_results;
        except Exception as e:
            raise CustomException(e,sys);

    def create_best_model(self,best_model,best_params,x_train,y_train,x_test,y_test):
        #### This fucntion will create the best model with best parameters
        logging.info(f'final model is getting createed for {best_model}')
        model=self.__models[best_model];
        model.set_params(**best_params)
        model.fit(x_train,y_train)
        y_train_pred=model.predict(x_train)
        y_test_pred=model.predict(x_test)

        training_result,test_result=self.__model_scoring(y_train=y_train,y_train_pred=y_train_pred,y_test=y_test,y_test_pred=y_test_pred)
        logging.info(f'Final Model performance calculation completed for {best_model} ')
        train_mse,train_mae,train_score=training_result
        test_mse,test_mae,test_score=test_result
        logging.info(f'Model performance data for {best_model} ')
        logging.info(f'Train MSE {train_mse} ')
        logging.info(f'Train MAE {train_mae} ')
        logging.info(f'Train Score {train_score} ')
        logging.info(f'Test MSE {test_mse} ')
        logging.info(f'Test MAE {test_mae} ')
        logging.info(f'Test Score {test_score} ')
        logging.info('Saving final model in pickle format')
        save_object(object=model,filename=self.__model_trainer_config.final_model_file_path,format='pickle')
        logging.info('saved final model in pickle format')


