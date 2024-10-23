import sys
import os
from src.logger.logging import logging
from src.exception.exception import custom_exception
from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifacts

class Train_Pipeline():
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def start_data_ingestion(self):
        logging.info("Entered the start data ingestion class")
        try:
            data_ingestion = DataIngestion(self.data_ingestion_config)
            data_ingestion_artifacts = data_ingestion.initiate_data_ingestion()

            logging.info("Got the train and validation from Gcloud storage")
            logging.info("Exited the start data ingestion class")
            return data_ingestion_artifacts
        except Exception as e:
            raise custom_exception(e, sys)
        
    
    def run_pipeline(self):
        logging.info("Entered  the run pipeline method of TrainPipeline class")

        try:
            data_ingestion_artifacts = self.start_data_ingestion()
            logging.info("Data ingestion is completed")
        except Exception as e:
            raise custom_exception(e, sys)