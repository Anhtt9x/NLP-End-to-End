import sys
import os
from src.logger.logging import logging
from src.exception.exception import custom_exception
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTranformation
from src.entity.config_entity import DataIngestionConfig, DataTranformationConfig ,ModelTrainerConfig,ModelEvaluationConfig
from src.entity.artifact_entity import DataIngestionArtifacts , DataTranformationArtifacts, ModelTrainerArtifacts, ModelEvaluationArtifacts
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation


class Train_Pipeline():
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTranformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()

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
        

    def start_data_transformation(self, data_ingestion_artifacts = DataIngestionArtifacts):
        logging.info("Entered the start data transformation class")
        try:
            data_transformation = DataTranformation(
                data_ingestion_artifacts=data_ingestion_artifacts,
                data_transformation_config=self.data_transformation_config
            )

            data_transformation_artifacts = data_transformation.initiate_data_transform()

            logging.info("Exited  the start data transformation class")
            return data_transformation_artifacts
        
        except Exception as e:
            raise custom_exception(e, sys)
        

    def start_model_trainer(self, data_transformation_artifacts = DataTranformationArtifacts):
        logging.info("Entered the start model trainer class")
        try:
            model_trainer = ModelTrainer(data_transformation_artifacts,
                                         self.model_trainer_config)
            
            model_trainer_artifacts = model_trainer.initiate_model_trainer()
            logging.info("Exited the start model trainer class")
            return model_trainer_artifacts
        except Exception as e:
            raise custom_exception(e, sys)
        
    
    def start_model_evaluation(self,model_trainer_artifacts:ModelTrainerArtifacts,
                               data_transformation_artifacts:DataTranformationArtifacts):
        logging.info("Entered the start model evaluation class")
        try:
            model_evaluation = ModelEvaluation(model_transformation_artifacts=data_transformation_artifacts,
                                               model_evaluation_config=self.model_evaluation_config,
                                               model_trainer_artifacts=model_trainer_artifacts)
            
            model_evaluation_artifacts = model_evaluation.initiate_model_evaluation()
            logging.info("Exited the start model evaluation class")
            return model_evaluation_artifacts
        except Exception as e:
            raise custom_exception(e,sys)

    
    def run_pipeline(self):
        logging.info("Entered  the run pipeline method of TrainPipeline class")

        try:
            data_ingestion_artifacts = self.start_data_ingestion()
            logging.info("Data ingestion is completed")

            data_transformation_artifacts = self.start_data_transformation(data_ingestion_artifacts)
            logging.info("Data transformation is completed")

            model_trainer_artifacts = self.start_model_trainer(data_transformation_artifacts)
            logging.info("Model trainer is completed")

            model_evaluation_artifacts = self.start_model_evaluation(model_trainer_artifacts,data_transformation_artifacts)
            logging.info("Model evaluation is completed")
        except Exception as e:
            raise custom_exception(e, sys)