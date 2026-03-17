import os;
import sys;
from src.logger import logging;
from src.exception import CustomException;
import pickle;
import pandas as pd;

def save_object(object,filename,format,mode='new'):
    try:
        logging.info('Starting data saving process')
        if format=='csv':
            object.to_csv(filename,index=False,header=True)
        elif format=='pickle':
            pickle.dump(object,open(filename,'wb'))
        elif format=='text' and mode=='append':
            with open(filename,'a') as file:
                file.writelines(str(object))
        elif format=='text' and mode=='write':
            with open(filename,'w') as file:
                file.writelines(str(object))
        logging.info('Data saving process Completed')
        

    except Exception as e:
        raise CustomException(e,sys);



