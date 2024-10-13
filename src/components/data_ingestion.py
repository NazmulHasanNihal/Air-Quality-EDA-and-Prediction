import os 
import sys
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import pandas as pd


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('data/processed', 'train.csv')
    test_data_path: str = os.path.join('data/processed', 'test.csv')
    raw_data_path: str = os.path.join('data/raw', 'GlobalWeatherRepository.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Started")
        try:
            df = pd.read_csv(self.ingestion_config.raw_data_path)
            logging.info(f"Raw dataset loaded from {self.ingestion_config.raw_data_path}")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            train_set, test_set =train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Train Test Split Completed")

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info(f"Train data saved to {self.ingestion_config.train_data_path}")
            logging.info(f"Test data saved to {self.ingestion_config.test_data_path}")
            
            logging.info("Data Ingestion Completed")
            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)

        except Exception as e:
            raise CustomException(e,sys)
        



if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    


