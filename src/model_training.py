import os
import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *
from utils.common_functions import read_yaml,load_data
from scipy.stats import randint

import mlflow
import mlflow.sklearn

logger = get_logger(__name__)

class ModelTraining:
    
    def __init__(self,train_path,test_path,model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path
        
        self.params_dist = LIGHTGM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS
        
    def load_and_split_data(self):
        try:
            logger.info(f"loading data from {self.train_path}")
            train_df = load_data(self.train_path)
            
            logger.info(f"loading data from {self.test_path}")
            test_df = load_data(self.test_path)
            
            X_train = train_df.drop(columns=['booking_status'])
            y_train = train_df['booking_status']
            
            X_test = test_df.drop(columns=['booking_status'])
            y_test = test_df['booking_status']
            
            logger.info(f"Data splitted successfully for model training")
            
            return X_train,y_train,X_test,y_test
        
        except Exception as e:
            logger.error(f"Error in loading and splitting data: {e}")
            raise CustomException(e)
        
    def train_lgbm(self,X_train,y_train):
        try:
            logger.info(f"Training LightGBM model")
            
            lgbm_model = lgb.LGBMClassifier(random_state=self.random_search_params['random_state'])
            
            logger.info(f"Performing RandomizedSearchCV for LightGBM model")
            
            random_search = RandomizedSearchCV(
                estimator=lgbm_model,
                param_distributions=self.params_dist,
                n_iter=self.random_search_params['n_iter'],
                cv=self.random_search_params['cv'],
                scoring=self.random_search_params['scoring'],
                n_jobs=self.random_search_params['n_jobs'],
                random_state=self.random_search_params['random_state'],
                verbose=self.random_search_params['verbose']
            )
            
            logger.info(f"Fitting RandomizedSearchCV")
            
            random_search.fit(X_train,y_train)
            
            logger.info("Hyperparameter tuning completed")
            
            best_params = random_search.best_params_
            best_lgbm_model = random_search.best_estimator_
            
            logger.info(f"Best parameters: {best_params}")
            
            return best_lgbm_model
            
        except Exception as e:
            logger.error(f"Error in training LightGBM model: {e}")
            raise CustomException(e)
        
    def evaluate_model(self,model,X_test,y_test):
        try:
            logger.info(f"Evaluating LightGBM model")
            
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test,y_pred)
            precision = precision_score(y_test,y_pred)
            recall = recall_score(y_test,y_pred)
            f1 = f1_score(y_test,y_pred)    
            
            logger.info(f"Accuracy: {accuracy}")
            logger.info(f"Precision: {precision}")
            logger.info(f"Recall: {recall}")
            logger.info(f"F1 Score: {f1}")
            
            return {
                "accuracy":accuracy,
                "precision":precision,
                "recall":recall,
                "f1":f1
            }
        except Exception as e:
            logger.error(f"Error in evaluating LightGBM model: {e}")
            raise CustomException(e)

    def save_model(self,model,model_output_path):
        try:
            logger.info(f"Saving LightGBM model")
            
            os.makedirs(os.path.dirname(self.model_output_path),exist_ok=True)
            
            joblib.dump(model,self.model_output_path)
            
            logger.info(f"Model saved successfully at {self.model_output_path}")
            
        except Exception as e:
            logger.error(f"Error in saving LightGBM model: {e}")
            raise CustomException(e)
        
    def run(self):
        try:
            logger.info("Starting model training")
            
            logger.info("starting our MLFLOW experiment")
            
            with mlflow.start_run() as run:
                logger.info(f"MLFLOW run started with ID: {run.info.run_id}")
                logger.info("logging the training and testing datasets to MLFLOW")
                
                mlflow.log_artifact(self.train_path,artifact_path="training_data")
                mlflow.log_artifact(self.test_path,artifact_path="testing_data")
                

                X_train,y_train,X_test,y_test = self.load_and_split_data()
                
                best_lgbm_model = self.train_lgbm(X_train,y_train)
                metrics = self.evaluate_model(best_lgbm_model,X_test,y_test)
                self.save_model(best_lgbm_model,self.model_output_path)
                logger.info("logging the model into MLFLOW")
                
                mlflow.sklearn.log_model(best_lgbm_model,artifact_path="best_lgbm_model")
                logger.info("logging parameters and metrics to MLFLOW")
                mlflow.log_params(best_lgbm_model.get_params())
                mlflow.log_metrics(metrics)

                mlflow.log_artifact(self.model_output_path,artifact_path="best_lgbm_model")

                logger.info(f"MLFLOW run completed with ID: {run.info.run_id}")
                
        except Exception as e:
            logger.error(f"Error in running MLFLOW experiment: {e}")
            raise CustomException(e)
        
if __name__ == "__main__":
    model_training = ModelTraining(PROCESSED_TRAIN_DATA_PATH,PROCESSED_TEST_DATA_PATH,MODEL_OUTPUT_PATH)
    model_training.run()
            
            
            



