import os;
import sys;
import pickle;
from src.exception import CustomException;

class PredictionPipeLine:
    def __init__(self):
        self.__model_path=os.path.join('artifacts','final_model.pkl');
        self.__transformer_path=os.path.join('artifacts','transformer_model.pkl')
        self.__model=pickle.load(open(self.__model_path,'rb'))
        self.__transformer=pickle.load(open(self.__transformer_path,'rb'))

    def model_prediction(self,input):
        try:
            transformed_input=self.__transformer.transform(input);
            print(transformed_input)
            prediction=self.__model.predict(transformed_input)
            return prediction
        except Exception as e:
            raise CustomException(e,sys);
