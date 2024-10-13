import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion


class DataTransformation:
    
    def __init__(self):
        self.transformed_train_path = 'data/processed/train.csv'
        self.transformed_test_path = 'data/processed/test.csv'

    def initiate_data_preprocessing(self, file_path):
        try:
            df = pd.read_csv(file_path)
            logging.info(f"Dataset loaded from {file_path}")

            df['last_updated'] = pd.to_datetime(df['last_updated'])
            df['sunrise'] = pd.to_datetime(df['sunrise'], format='%I:%M %p', errors='coerce')
            df['sunset'] = pd.to_datetime(df['sunset'], format='%I:%M %p', errors='coerce')
            df['moonrise'] = pd.to_datetime(df['moonrise'], format='%I:%M %p', errors='coerce')
            df['moonset'] = pd.to_datetime(df['moonset'], format='%I:%M %p', errors='coerce')
            logging.info("Time data is converted to datetime format")

            df = df.drop(columns=[
                'temperature_fahrenheit', 'feels_like_fahrenheit', 'wind_mph', 'pressure_in', 
                'precip_in', 'visibility_miles'
            ])
            logging.info("Redundant columns are dropped")

            df = df.dropna(subset=[
                'air_quality_Carbon_Monoxide', 'air_quality_Ozone', 'air_quality_Nitrogen_dioxide',
                'air_quality_Sulphur_dioxide', 'air_quality_PM2.5', 'air_quality_PM10'
            ])
            logging.info("Missing values are dropped")


            df['air_quality_us-epa-index'] = df['air_quality_us-epa-index'].astype('Int64')
            df['air_quality_gb-defra-index'] = df['air_quality_gb-defra-index'].astype('Int64')
            logging.info("Column types are converted to Int64")

            logging.info("Data Preprocessing Completed")

            return df

        except Exception as e:
            raise CustomException(e, sys)

    def apply_data_transformation(self, train_file_path, test_file_path):
        logging.info("Data Transformation Started")
        try:
            train_df = self.initiate_data_preprocessing(train_file_path)
            test_df = self.initiate_data_preprocessing(test_file_path)
            logging.info('Preprocessed both training and test datasets')

            numerical_cols = [
                'temperature_celsius', 'wind_kph', 'pressure_mb', 'precip_mm', 'humidity',
                'cloud', 'feels_like_celsius', 'visibility_km', 'uv_index', 'gust_kph',
                'air_quality_Carbon_Monoxide', 'air_quality_Ozone', 'air_quality_Nitrogen_dioxide',
                'air_quality_Sulphur_dioxide', 'air_quality_PM2.5', 'air_quality_PM10'
            ]
            categorical_cols = ['condition_text']

            preprocessor = ColumnTransformer(
                transformers=[
                    ('scaler', StandardScaler(), numerical_cols),
                    ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols)
                ]
            )

            pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

            transformed_train_data = pipeline.fit_transform(train_df)
            transformed_test_data = pipeline.transform(test_df)
            logging.info('Transformed both training and test data')

            onehotencoder = pipeline.named_steps['preprocessor'].transformers_[1][1]
            onehot_encoded_cols = list(onehotencoder.get_feature_names_out(categorical_cols))

            all_columns = numerical_cols + onehot_encoded_cols


            transformed_train_df = pd.DataFrame(transformed_train_data, columns=all_columns)
            transformed_test_df = pd.DataFrame(transformed_test_data, columns=all_columns)
            logging.info('Created DataFrames with transformed columns')


            transformed_train_df.to_csv(self.transformed_train_path, index=False)
            transformed_test_df.to_csv(self.transformed_test_path, index=False)
            logging.info(f"Transformed training data saved to {self.transformed_train_path}")
            logging.info(f"Transformed test data saved to {self.transformed_test_path}")

            logging.info("Data Transformation Completed")

            return self.transformed_train_path, self.transformed_test_path

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformer = DataTransformation()
    data_transformer.apply_data_transformation(train_data, test_data)
