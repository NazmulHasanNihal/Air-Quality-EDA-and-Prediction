from src.components.data_ingestion import DataIngestion
from src.exception import CustomException
from src.logger import logging
import os 
import sys
import pandas as pd

class DataPreprocessing:
    def __init__(self):
        pass

    def initiate_data_preprocessing(self):
        try:
            pre_df = DataIngestion().initiate_data_ingestion()
            logging.info("Dataset is loaded")

            pre_df['last_updated'] = pd.to_datetime(pre_df['last_updated'])
            pre_df['sunrise'] = pd.to_datetime(pre_df['sunrise'], format='%I:%M %p', errors='coerce')
            pre_df['sunset'] = pd.to_datetime(pre_df['sunset'], format='%I:%M %p', errors='coerce')
            pre_df['moonrise'] = pd.to_datetime(pre_df['moonrise'], format='%I:%M %p', errors='coerce')
            pre_df['moonset'] = pd.to_datetime(pre_df['moonset'], format='%I:%M %p', errors='coerce')
            logging.info("Time data is converted to datetime format")

            pre_df = pre_df.drop(columns=[
                'temperature_fahrenheit', 'feels_like_fahrenheit', 'wind_mph', 'pressure_in', 'precip_in', 'visibility_miles'
            ])
            logging.info("Redundant Columns are dropped")

            pre_df = pre_df.dropna(subset=[
                'air_quality_Carbon_Monoxide', 'air_quality_Ozone', 'air_quality_Nitrogen_dioxide',
                'air_quality_Sulphur_dioxide', 'air_quality_PM2.5', 'air_quality_PM10'
            ])
            logging.info("Missing values are dropped")

            pre_df['air_quality_us-epa-index'] = pre_df['air_quality_us-epa-index'].astype('Int64')
            pre_df['air_quality_gb-defra-index'] = pre_df['air_quality_gb-defra-index'].astype('Int64')
            logging.info("Column types are converted to int64")

            pre_df.to_csv('data/processed/preprocessed_GWR.csv', index=False)
            logging.info("Dataset is saved as preprocessed_GWR.csv")
            
            logging.info("Data Preprocessing Completed")
            return pre_df
            
            
        except Exception as e:
            raise CustomException(e,sys)
            

if __name__ == "__main__":
    obj = DataPreprocessing()
    obj.initiate_data_preprocessing()

        
    