import os;
import sys;
from src.exception import CustomException;
from src.logger import logging;
import pandas as pd;
from sklearn.model_selection import train_test_split;
from dataclasses import dataclass;
from src.utils import save_object;
from typing import Tuple;

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw_data.csv')


class DataIngestion:
    def __init__(self):
        self.__ingestion_config=DataIngestionConfig();
    def initiate_data_ingestion(self,filename) -> Tuple[str,str]:
        logging.info('Entered the data ingestion method')
        try:
            df=pd.read_csv(filename)
            logging.info('Data read successfull')

            os.makedirs(os.path.dirname(self.__ingestion_config.train_data_path),exist_ok=True)
            #df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            save_object(object=df,filename=self.__ingestion_config.raw_data_path,format='csv')
            logging.info('Raw data saved')
            logging.info('Train Test split started')
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            #train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            save_object(object=train_set,filename=self.__ingestion_config.train_data_path,format='csv')
            #test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            save_object(object=test_set,filename=self.__ingestion_config.test_data_path,format='csv')
            logging.info('Train test split completed and train test data is saved')

            return (self.__ingestion_config.train_data_path,
                    self.__ingestion_config.test_data_path
                    )
        except Exception as e:
            raise CustomException(e,sys)

# if __name__=='__main__':
#     obj=DataIngestion();
#     obj.initiate_data_ingestion()