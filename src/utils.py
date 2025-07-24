import sys
import pandas as pd
import numpy as np
import dill
import os
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True) 
        
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    

def model_evaluate(X_train,Y_train,X_test,Y_test,models,param):
    try:
        report={}

        for model_name in models:
            model = models[model_name]
            params = param[model_name]
            
            if params:
                # GridSearchCV for hyperparameter tuning
                gs = GridSearchCV(
                    estimator=model,
                    param_grid=params,
                    cv=3,
                )
                
                # Fit GridSearchCV
                gs.fit(X_train, Y_train)
                
                # Get best model
                best_model = gs.best_estimator_
                best_params = gs.best_params_
            else:
                # No parameters to tune, use default model
                model.fit(X_train, Y_train)
                best_model = model
                best_params = {}
            
            # Make predictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)
            
            # Calculate scores
            train_score = r2_score(Y_train, y_train_pred)
            test_score = r2_score(Y_test, y_test_pred)
            
            report[model_name] = {
                'train_score': train_score,
                'test_score': test_score,
                'best_model': best_model,
                'best_params': best_params
            }

        return report
    except Exception as e:
        raise CustomException(e,sys)