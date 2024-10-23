from zipfile import ZipFile
import os
import sys
from src.logger.logging import logging
from src.exception.exception import custom_exception
from src.configuration.gcloud_syncer import GcloudSync
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifacts

class DataIngestion:
    def __init__(self, data_ingestion_config:DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config
        self.gcloud = GcloudSync()

    def get_data_from_gcloud(self):
        try:    
            logging.info("Entered the get data from cloud method of Data Ingestion class")
            os.makedirs(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR,exist_ok=True)
            self.gcloud.sync_folder_from_cloud(self.data_ingestion_config.BUCKET_NAME,
                                               self.data_ingestion_config.ZIP_FILE_NAME,
                                               self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR)
            
            logging.info("Exited the data from cloud")
        
        except Exception as e:
            raise custom_exception(e,sys)
        

    def unzip_and_clean(self):
        logging.info("Entered the unzip_and_clean method of Data ingestion class")
        try: 
            with ZipFile(self.data_ingestion_config.ZIP_FILE_PATH, 'r') as zip_ref:
                zip_ref.extractall(self.data_ingestion_config.ZIP_FILE_DIR)

            logging.info("Exited the unzip_and_clean method of Data ingestion class")

            return self.data_ingestion_config.DATA_ARTIFACTS_DIR, self.data_ingestion_config.DATA_NEW_ARTIFACTS_DIR

        except Exception as e:
            raise custom_exception(e, sys)
    
    def  initiate_data_ingestion(self):
        logging.info("Entered the initiate_data_ingestion")

        try:
            self.get_data_from_gcloud()
            logging.info("Fetch data from gcloud bucket")
            imbalance_data_file_path , raw_data_file_path = self.unzip_and_clean()
            logging.info("Unzipped the data and split into train and validation class")
            data_ingestion_artifacts = DataIngestionArtifacts(imbalance_data_file_path,raw_data_file_path)

            logging.info("Exited the initiate_data_ingestion method")
            logging.info(f"Data ingestion artifact: {data_ingestion_artifacts}")

            return data_ingestion_artifacts

        except Exception as e:
            raise custom_exception(e,sys)
        