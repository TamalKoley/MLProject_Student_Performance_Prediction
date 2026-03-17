import os;
import sys;
from src.exception import CustomException;
from src.logger import logging;
from src.components.data_ingestion import DataIngestion;
from src.components.data_transformation import DataTransformation;
from src.components.model_trainer import ModelTrainer;


if __name__=='__main__':
    try:
        logging.info('Main Program Started');
        data_ing=DataIngestion();
        train_datapath,test_data_path=data_ing.initiate_data_ingestion('notebook\data\stud.csv')
        data_trasnform=DataTransformation();
        x_train,y_train,x_test,y_test,_=data_trasnform.transform_data(train_data_path=train_datapath,test_data_path=test_data_path,target_name='math_score')
        model_trainer=ModelTrainer();
        best_models=model_trainer.initiate_model_training(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test)
        tunning_result=model_trainer.initiate_hyperparameter_tunning(best_models=best_models,x_train=x_train,y_train=y_train)
        best_score=-9999;
        best_model_name='';
        best_model_params={};
        for model in tunning_result:
                if model['best_score']> best_score:
                     best_score=model['best_score']
                     best_model_name=model['model_name']
                     best_model_params=model['best_params']

        model_trainer.create_best_model(
            best_model=best_model_name,
            best_params=best_model_params,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test
            )
        logging.info('Main Program Completed');


    except Exception as e:
        try:
            raise CustomException(e,sys);
        except CustomException as error:
            logging.info(error.error_message)
