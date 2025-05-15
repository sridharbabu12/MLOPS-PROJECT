import os
import pandas as pd
import boto3
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from io import StringIO
from utils.common_functions import read_yaml
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *


load_dotenv(dotenv_path=os.path.join("config",".env"))

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self,config):
        self.config=config["data_ingestion"]
        self.bucket=self.config['bucket']
        self.key=self.config['key']
        self.train_test_ratio=self.config['train_ratio']
        
        os.makedirs(RAW_DIR,exist_ok=True)
        
        logger.info(f"data ingestion started with {self.bucket} and file is {self.key}")
        
    def download_csv_from_s3(self):
        try:
            s3 = boto3.client(
                's3',
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=os.getenv("AWS_REGION")
            )
            bucket = "mlopsproject01"
            key = "Hotel Reservations.csv"
            
            s3.download_file(bucket,key,RAW_FILE_PATH)
            logger.info(f"CSV file is successfully downloaded to {RAW_FILE_PATH}")
            
        except Exception as e:
            logger.error("error while downloding the csv file")
            raise CustomException("failed to download csv file")
        
    def split_data(self):
        try:
            logger.info("starting the splitting process")
            data= pd.read_csv(RAW_FILE_PATH)
            train_data , test_data = train_test_split(data,test_size=1-self.train_test_ratio,random_state=42)
            
            train_data.to_csv(TRAIN_FILE_PATH)
            test_data.to_csv(TEST_FILE_PATH)
            
            logger.info(f"train data saved to {TRAIN_FILE_PATH}")
            logger.info(f"test data saved to {TEST_FILE_PATH}")
            
        except Exception as e:
            logger.error("error while splitting the data")
            raise CustomException("failed to split data into training and test sets")
        
    def run(self):
        try:
            logger.info("starting data ingestion process")
            
            self.download_csv_from_s3()
            self.split_data()
            
            logger.info("data ingestion completed successfully")
            
        except CustomException as ce:
            logger.error(f"CustomException : {str(ce)}")
            
        finally:
            logger.info("Data ingestion completed")
            
if __name__=="__main__":
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()
            
            

            
        








