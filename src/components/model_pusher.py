import sys
from src.logger.logging import logging
from src.exception.exception import custom_exception
from src.configuration.gcloud_syncer import GcloudSync
from src.entity.artifact_entity import ModelPusherArtifacts
from src.entity.config_entity import ModelPusherConfig


class ModelPusher:
    def __init__(self,model_pusher_config: ModelPusherConfig):
       
        self.model_pusher_config =  model_pusher_config
        self.gcloud_sync = GcloudSync()


    def initiate_model_pusher(self) -> ModelPusherArtifacts:
        logging.info("Entered initiate model pusher method of ModelTrainer class")
        try:
            self.gcloud_sync.sync_folder_to_gcloud(self.model_pusher_config.BUCKET_NAME,
                                                   self.model_pusher_config.TRAINED_MODEL_PATH,
                                                   self.model_pusher_config.MODEL_NAME)
            
            logging.info("Upload the best model to gcloud storage")

            model_pusher_artifact = ModelPusherArtifacts(
                bucket_name=self.model_pusher_config.BUCKET_NAME
            )

            logging.info("Exited the initiate model pusher method of ModelTrainer class")

            return model_pusher_artifact

        except Exception as e:
            raise custom_exception(e,sys)
        
                