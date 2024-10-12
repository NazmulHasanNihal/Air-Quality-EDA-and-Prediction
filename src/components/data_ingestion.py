import os 
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd



class DataIngestion:
    def __init__(self):
        pass

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Started")
        try:
            df = pd.read_csv(r'data/raw/GlobalWeatherRepository.csv')
            logging.info("Data Ingestion Completed")
            return df
        
        except Exception as e:
            raise CustomException(e,sys)



if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
