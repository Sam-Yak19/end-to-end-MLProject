import sys
import pandas as pd
import numpy as np
import dill
import os
from src.exception import CustomException
from sklearn.metrics import r2_score

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True) 
        
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    

def model_evaluate(X_train,Y_train,X_test,Y_test,models):
    try:
        report={}

        for model_name, model in models.items():
            # Train model
            model.fit(X_train, Y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate scores
            train_score = r2_score(Y_train, y_train_pred)
            test_score = r2_score(Y_test, y_test_pred)
            
            report[model_name] = {
                'train_score': train_score,
                'test_score': test_score
            }

        return report
    except Exception as e:
        raise CustomException(e,sys)