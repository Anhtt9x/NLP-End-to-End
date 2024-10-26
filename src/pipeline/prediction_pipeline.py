import os
import sys
import io
import keras
import pickle
from PIL import Image
from src.logger.logging import logging
from src.exception.exception import custom_exception
from src.constants import *
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from src.configuration.gcloud_syncer import GcloudSync
from src.components.data_transformation import DataTranformation
from src.entity.artifact_entity import DataIngestionArtifacts
from src.entity.config_entity import DataTranformationConfig


class PredictionPipeline:
    def __init__(self):
        self.bucket_name = BUCKET_NAME
        self.model_name = MODEL_NAME
        self.model_path = os.path.join("artifacts","PredictModel")
        self.gcloud = GcloudSync()
        self.data_transformation = DataTranformation(data_transformation_config=DataTranformationConfig,
                                                     data_ingestion_artifacts=DataIngestionArtifacts)
        
    def get_model_from_gcloud(self):
        try:
            logging.info("Entered the get model from  gcloud function")
            os.makedirs(self.model_path,exist_ok=True)
            self.gcloud.sync_folder_from_cloud(self.bucket_name,self.model_name,self.model_path)

            best_model_path = os.path.join(self.model_path,self.model_name)
            logging.info("Exited  the get model from gcloud function")
            return best_model_path
        
        except Exception as e:
            raise custom_exception(e,sys)
        
    
    def predict(self,best_model_path,text):
        logging.info("Running the predict function")
        try:
            best_model_path:str = self.get_model_from_gcloud()
            load_model = keras.models.load_model(best_model_path)
            with open("tokenizer.pickle","rb") as handle:
                tokenizer = pickle.load(handle)

            text = self.data_transformation.concat_data_cleaning(text)
            print(text)

            sequences = tokenizer.texts_to_sequences(text)
            padded_sequences = pad_sequences(sequences,maxlen=300)
            prediction = load_model.predict(padded_sequences)
            print(f"prediction: {prediction}")

            if prediction > 0.5:
                print("hate and abusive")
                return "hate and abusive"
            
            else:
                print("no hate")
                return "no hate"
            
        except Exception as e:
            raise custom_exception(e,sys)



    def run_pipeline(self,text):
        logging.info("Entered the run pipeline method of PredictionPipeline function")
        try:
            best_model_path:str = self.get_model_from_gcloud()
            predict_text = self.predict(best_model_path,text)
            logging.info("Exited the run pipeline method of PredictionPipeLine class")
            return predict_text
        except Exception as e:
            raise custom_exception(e,sys)