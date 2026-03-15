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
        print(best_models)
        logging.info('Main Program Completed');
    except Exception as e:
        raise CustomException(e,sys);
