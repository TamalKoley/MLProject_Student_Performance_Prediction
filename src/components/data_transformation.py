import os;
import sys;
from src.exception import CustomException;
from src.logger import logging;
import pandas as pd;
from dataclasses import dataclass;
from sklearn.preprocessing import StandardScaler,OneHotEncoder;
from src.components.data_ingestion import DataIngestion;
from sklearn.compose import ColumnTransformer;
from sklearn.pipeline import Pipeline;
from sklearn.impute import SimpleImputer;
import pickle;
import numpy as np;

@dataclass
class DataTransformation:
    def __init__(self):
        self.transformer_model_path=os.path.join('artifacts','transformer_model.pkl')
    def __get_transformer(self,numerical_columns,categorical_columns):
        try:
            logging.info('Starting transformer creation')
            numeric_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('standard_scalar',StandardScaler())
                ]
            )
            categorical_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('One_Hot',OneHotEncoder(drop='first'))
                ]
            )
            transformer=ColumnTransformer(
                [
                    ('numeric_pipeline',numeric_pipeline,numerical_columns),
                    ('categorical_pipeline',categorical_pipeline,categorical_columns)
                ]
            )
            logging.info('Transformer created successfully')
            return transformer
        except Exception as e:
            raise CustomException(e,sys);
    
    def __save_transformer(self,transformer_obj):
        pickle.dump(transformer_obj,open(self.transformer_model_path,'wb'))

    def transform_data(self,train_data_path,test_data_path,target_name):
        try:
            logging.info('Starting Data read for transformation')
            train_dataset=pd.read_csv(train_data_path)
            test_dataset=pd.read_csv(test_data_path)
            logging.info('Completed Data read for transformation')
            numerical_columns=[column for column in train_dataset.columns if train_dataset[column].dtype!='O' and column!=target_name]
            categorical_columns=[column for column in train_dataset.columns if train_dataset[column].dtype=='O' and column!=target_name]
            logging.info('Numerical and categorical columns are created')
            transformer_obj=self.__get_transformer(numerical_columns=numerical_columns,categorical_columns=categorical_columns)
            x_train_dataset=train_dataset.drop(columns=[target_name],axis=1)
            x_test_dataset=test_dataset.drop(columns=[target_name],axis=1)
            logging.info('Starting fit transform ')
            x_train_dataset_transformed=transformer_obj.fit_transform(x_train_dataset)
            x_test_dataset_transformed=transformer_obj.transform(x_test_dataset)
            logging.info('completed fit transform ')
            train_arr=np.c_[x_train_dataset_transformed,np.array(x_train_dataset)]
            test_arr=np.c_[x_test_dataset_transformed,np.array(x_test_dataset)]
            self.__save_transformer(transformer_obj)
            logging.info('Transformer Object saved')
            return(
                train_arr,
                test_arr,
                self.transformer_model_path
            )
            

        except Exception as e:
            raise CustomException(e,sys)

if __name__=='__main__':
    dt=DataTransformation()
    di=DataIngestion()
    train_path,test_path=di.initiate_data_ingestion();
    test_array,train_array,model_path=dt.transform_data(train_data_path=train_path,test_data_path=test_path,target_name='math_score')



