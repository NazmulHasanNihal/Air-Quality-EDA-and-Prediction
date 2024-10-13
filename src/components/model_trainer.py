import os
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from dataclasses import dataclass
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostRegressor, BaggingClassifier, ExtraTreesClassifier, GradientBoostingRegressor, RandomForestRegressor,
    AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
)
from sklearn.linear_model import (
    LinearRegression, Perceptron, Ridge, Lasso, ElasticNet, LogisticRegression, SGDClassifier
)
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.metrics import r2_score, accuracy_score
from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("models","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array, task_type='regression'):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            if task_type == 'regression':
                # Regression Models
                models = {
                    "Random Forest": RandomForestRegressor(),
                    "Decision Tree": DecisionTreeRegressor(),
                    "Gradient Boosting": GradientBoostingRegressor(),
                    "Linear Regression": LinearRegression(),
                    "Ridge Regression": Ridge(),
                    "Lasso Regression": Lasso(),
                    "ElasticNet Regression": ElasticNet(),
                    "K-Neighbors Regressor": KNeighborsRegressor(),
                    "XGBRegressor": XGBRegressor(),
                    "CatBoost Regressor": CatBoostRegressor(verbose=False),
                    "AdaBoost Regressor": AdaBoostRegressor(),
                    "SVR": SVR(),
                    "LGBM Regressor": LGBMRegressor()
                }

                params = {
                    "Decision Tree": {
                        'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                    },
                    "Random Forest": {
                        'n_estimators': [8, 16, 32, 64, 128, 256]
                    },
                    "Gradient Boosting": {
                        'learning_rate': [0.1, 0.01, 0.05, 0.001],
                        'n_estimators': [8, 16, 32, 64, 128, 256]
                    },
                    "Linear Regression": {},
                    "Ridge Regression": {
                        'alpha': [0.1, 1, 10, 100]
                    },
                    "Lasso Regression": {
                        'alpha': [0.1, 1, 10, 100]
                    },
                    "ElasticNet Regression": {
                        'alpha': [0.1, 1, 10, 100],
                        'l1_ratio': [0.1, 0.5, 0.7, 1]
                    },
                    "K-Neighbors Regressor": {
                        'n_neighbors': [3, 5, 7, 9]
                    },
                    "XGBRegressor": {
                        'learning_rate': [0.1, 0.01, 0.05, 0.001],
                        'n_estimators': [8, 16, 32, 64, 128, 256]
                    },
                    "CatBoost Regressor": {
                        'depth': [6, 8, 10],
                        'learning_rate': [0.01, 0.05, 0.1],
                        'iterations': [30, 50, 100]
                    },
                    "AdaBoost Regressor": {
                        'learning_rate': [0.1, 0.01, 0.5, 0.001],
                        'n_estimators': [8, 16, 32, 64, 128, 256]
                    },
                    "SVR": {
                        'kernel': ['linear', 'rbf'],
                        'C': [0.1, 1, 10]
                    },
                    "LGBM Regressor": {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.05, 0.1]
                    }
                }
            else:
                # Classification Models
                models = {
                    "Random Forest": RandomForestClassifier(),
                    "Decision Tree": DecisionTreeClassifier(),
                    "Gradient Boosting": GradientBoostingClassifier(),
                    "Logistic Regression": LogisticRegression(),
                    "K-Neighbors Classifier": KNeighborsClassifier(),
                    "XGBClassifier": XGBClassifier(),
                    "CatBoost Classifier": CatBoostClassifier(verbose=False),
                    "AdaBoost Classifier": AdaBoostClassifier(),
                    "SVM": SVC(),
                    "LGBM Classifier": LGBMClassifier(),
                    "Naive Bayes": GaussianNB(),
                    "LDA": LinearDiscriminantAnalysis(),
                    "QDA": QuadraticDiscriminantAnalysis(),
                    "Perceptron": Perceptron(),
                    "SGD": SGDClassifier(),
                    "Extra Trees Classifier": ExtraTreesClassifier(),
                    "Bagging Classifier": BaggingClassifier()
                }

                params = {
                    "Decision Tree": {
                        'criterion': ['gini', 'entropy']
                    },
                    "Random Forest": {
                        'n_estimators': [8, 16, 32, 64, 128, 256]
                    },
                    "Gradient Boosting": {
                        'learning_rate': [0.1, 0.01, 0.05, 0.001],
                        'n_estimators': [8, 16, 32, 64, 128, 256]
                    },
                    "Logistic Regression": {
                        'C': [0.1, 1, 10]
                    },
                    "K-Neighbors Classifier": {
                        'n_neighbors': [3, 5, 7, 9]
                    },
                    "XGBClassifier": {
                        'learning_rate': [0.1, 0.01, 0.05, 0.001],
                        'n_estimators': [8, 16, 32, 64, 128, 256]
                    },
                    "CatBoost Classifier": {
                        'depth': [6, 8, 10],
                        'learning_rate': [0.01, 0.05, 0.1],
                        'iterations': [30, 50, 100]
                    },
                    "AdaBoost Classifier": {
                        'learning_rate': [0.1, 0.01, 0.5, 0.001],
                        'n_estimators': [8, 16, 32, 64, 128, 256]
                    },
                    "SVM": {
                        'kernel': ['linear', 'rbf'],
                        'C': [0.1, 1, 10]
                    },
                    "LGBM Classifier": {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.05, 0.1]
                    }
                }

            # Evaluate all models
            model_report = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)

            # Find the best model
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(f"Best model found: {best_model_name} with score {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Predict using the best model
            predicted = best_model.predict(X_test)

            if task_type == 'regression':
                score = r2_score(y_test, predicted)
            else:
                score = accuracy_score(y_test, predicted)

            return score

        except Exception as e:
            raise CustomException(e, sys)

