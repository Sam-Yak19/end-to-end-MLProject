import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import model_evaluate

@dataclass
class ModelTrainerConfig:
    model_trainer_obj_file_path=os.path.join("artifacts","model.pkl")

class Modeltrainer:
    def __init__(self):
        self.modeltrainerconfig=ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split Training and test dataset")
            X_train,Y_train,X_test,Y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                'LinearRegression': LinearRegression(),
                'Ridge': Ridge(),
                'Lasso': Lasso(),
                'KNeighborsRegressor': KNeighborsRegressor(),
                'SVR': SVR(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'RandomForestRegressor': RandomForestRegressor(),
                'AdaBoostRegressor': AdaBoostRegressor(),
                'GradientBoostingRegressor': GradientBoostingRegressor(),
                'CatBoostRegressor': CatBoostRegressor(verbose=False),
                'XGBRegressor': XGBRegressor()
            }

            model_report=model_evaluate(X_train=X_train,
                Y_train=Y_train,
                X_test=X_test,
                Y_test=Y_test,
                models=models)

            best_model_score=max([score['test_score'] for score in model_report.values()])

            best_model_name=None
            for model_name, scores in model_report.items():
                if scores['test_score'] == best_model_score:
                    best_model_name = model_name
                    break

            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("There is no best mdoel found")
            logging.info("Best model found on both training and testing datatset")

            save_object(
                file_path=self.modeltrainerconfig.model_trainer_obj_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            predicted_value=r2_score(predicted,Y_test)

            return predicted_value
        
        except Exception as e:
            raise CustomException(e,sys)
