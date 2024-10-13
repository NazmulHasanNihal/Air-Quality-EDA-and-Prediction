import os
import sys
import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, accuracy_score
from src.exception import CustomException
from src.logger import logging

# 1. Save a Python object to a file
def save_object(file_path, obj):
    """
    Save a Python object to a file using pickle.
    Args:
        file_path (str): The file path where the object will be saved.
        obj (Any): The object to be saved.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Object saved at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)

# 2. Load a Python object from a file
def load_object(file_path):
    """
    Load a Python object from a file using pickle.
    Args:
        file_path (str): The file path where the object is saved.
    Returns:
        The loaded object.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

# 3. Evaluate models with GridSearchCV
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Trains and evaluates multiple models with hyperparameter tuning.
    
    Args:
        X_train (np.array): Training features.
        y_train (np.array): Training target.
        X_test (np.array): Test features.
        y_test (np.array): Test target.
        models (dict): Dictionary of models to be evaluated.
        param (dict): Dictionary of hyperparameters for each model.
    
    Returns:
        dict: Model name as key and evaluation score as value.
    """
    try:
        model_report = {}
        
        for model_name, model in models.items():
            logging.info(f"Evaluating model: {model_name}")
            params = param.get(model_name, {})

            # Perform Grid Search CV for hyperparameter tuning
            gs = GridSearchCV(model, params, cv=3, n_jobs=-1, verbose=2)
            gs.fit(X_train, y_train)
            
            best_model = gs.best_estimator_
            logging.info(f"Best parameters for {model_name}: {gs.best_params_}")

            # Make predictions
            y_pred = best_model.predict(X_test)

            # Calculate score (R2 for regression, accuracy for classification)
            if len(np.unique(y_test)) > 2:  # Assuming regression if continuous target
                score = r2_score(y_test, y_pred)
                logging.info(f"{model_name} - R2 score: {score}")
            else:
                score = accuracy_score(y_test, y_pred)
                logging.info(f"{model_name} - Accuracy score: {score}")

            model_report[model_name] = score

        return model_report

    except Exception as e:
        raise CustomException(e, sys)

# 4. Calculate R2 score for regression
def calculate_r2_score(y_true, y_pred):
    """
    Calculates the R2 score for a regression model.
    
    Args:
        y_true (np.array): True target values.
        y_pred (np.array): Predicted target values.
    
    Returns:
        float: R2 score.
    """
    try:
        return r2_score(y_true, y_pred)
    except Exception as e:
        raise CustomException(e, sys)

# 5. Calculate accuracy score for classification
def calculate_accuracy(y_true, y_pred):
    """
    Calculates the accuracy score for a classification model.
    
    Args:
        y_true (np.array): True target values.
        y_pred (np.array): Predicted target values.
    
    Returns:
        float: Accuracy score.
    """
    try:
        return accuracy_score(y_true, y_pred)
    except Exception as e:
        raise CustomException(e, sys)
