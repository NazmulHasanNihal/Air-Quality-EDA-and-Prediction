import os 
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_preprocessing import DataPreprocessing
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataTransformation:

    def __init__(self):
        pass

    def get_transformer_object(self):
        try:

            transformed_df = DataPreprocessing().initiate_data_preprocessing()
            logging.info('Loaded the preprocessed dataset')

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

            # Create a pipeline
            pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

            # Apply the transformations
            transformed_data = pipeline.fit_transform(transformed_df)
            logging.info('Transformed numerical and categorical columns')

  
            onehotencoder = pipeline.named_steps['preprocessor'].transformers_[1][1]
            onehot_encoded_cols = list(onehotencoder.get_feature_names_out(categorical_cols))


            all_columns = numerical_cols + onehot_encoded_cols


            transformed_df = pd.DataFrame(transformed_data, columns=all_columns)
            logging.info('Created DataFrame with transformed columns')


            transformed_df['temp_range'] = transformed_df['temperature_celsius'] - transformed_df['feels_like_celsius']
            logging.info('Created new feature "temp_range" based on existing ones')

            moon_phase_mapping = {
                'New Moon': 0, 'Waxing Crescent': 1, 'First Quarter': 2, 'Waxing Gibbous': 3,
                'Full Moon': 4, 'Waning Gibbous': 5, 'Last Quarter': 6, 'Waning Crescent': 7
            }
            logging.info('Created mapping for moon phase')

            if 'moon_phase' in transformed_df.columns:
                transformed_df['moon_phase'] = transformed_df['moon_phase'].map(moon_phase_mapping)
                logging.info('Mapped moon phase column')

            
            columns_to_drop = ['country', 'location_name', 'timezone', 'sunrise', 'sunset', 'moonrise', 'moonset']
            existing_columns_to_drop = [col for col in columns_to_drop if col in transformed_df.columns]

           
            transformed_df = transformed_df.drop(columns=existing_columns_to_drop)
            logging.info(f"Dropped unnecessary columns: {existing_columns_to_drop}")

            
            transformed_df.to_csv('data/processed/transformed_GWR.csv', index=False)
            logging.info('Saved the transformed dataset as transformed_GWR.csv')


            logging.info('Data Transformation completed')
            return transformed_df

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataTransformation()
    obj.get_transformer_object()
