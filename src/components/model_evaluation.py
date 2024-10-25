import os
import sys
import keras
import pickle
import numpy as np
import pandas as pd
from src.logger.logging import logging
from src.exception.exception import custom_exception
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from src.constants import *
from src.configuration.gcloud_syncer import GcloudSync
from sklearn.metrics import confusion_matrix
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelEvaluationArtifacts, ModelTrainerArtifacts,DataTranformationArtifacts

class ModelEvaluation:
    def __init__(self,model_evaluation_config:ModelEvaluationConfig,
                 model_trainer_artifacts:ModelTrainerArtifacts,
                 model_transformation_artifacts: DataTranformationArtifacts):
        
        self.model_evaluation_config = model_evaluation_config
        self.model_trainer_artifacts = model_trainer_artifacts
        self.model_transformation_artifacts = model_transformation_artifacts

        self.gcloud = GcloudSync()
        
    
    def get_best_model_from_gcloud(self):
        try:
            logging.info("Entered the get best model from gcloud method of Model Evaluation class")
            os.makedirs(self.model_evaluation_config.MODEL_EVALUATION_FILE_PATH,exist_ok=True)

            self.gcloud.sync_folder_from_cloud(self.model_evaluation_config.BUCKET_NAME,
                                               self.model_evaluation_config.MODEL_EVALUATION_FILE_NAME,
                                               self.model_evaluation_config.MODEL_EVALUATION_FILE_PATH)
            
            best_model_path = os.path.join(self.model_evaluation_config.MODEL_EVALUATION_FILE_PATH,
                                           self.model_evaluation_config.MODEL_EVALUATION_FILE_NAME)

            logging.info("Exited the best model from gcloud method of Model Evaluation class")
            return best_model_path
        
        except Exception as e:
            raise custom_exception(e,sys)
        
    def evaluate(self):
        try:
            logging.info("Entering into to the evaluate  method of Model Evaluation class")
            print(self.model_trainer_artifacts.x_test_path)

            x_test = pd.read_csv(self.model_trainer_artifacts.x_test_path,index_col=0)
            y_test = pd.read_csv(self.model_trainer_artifacts.y_test_path,index_col=0)

            with open("tokenizer.pickle","rb") as handle:
                tokenizer = pickle.load(handle)
            
            load_model = keras.models.load_model(self.model_trainer_artifacts.trained_model_path)
            x_test = x_test['tweet'].astype(str)
            x_test = x_test.squeeze()
            y_test = y_test.squeeze()

            test_squences = tokenizer.texts_to_squences(x_test)
            test_squences_matrix = pad_sequences(test_squences,maxlen=MAX_LEN)

            print(f"-------------{test_squences_matrix}-----------------")
            print(f"-------------{x_test.shape}-------------------------")

            accuracy = load_model.evaluate(test_squences_matrix,y_test)
            logging.info(f"The test accuracy {accuracy}")

            lstm_prediction = load_model.predict(test_squences_matrix)

            res = []
            for prediction in lstm_prediction:
                if prediction > 0.5:
                    res.append(1)
                else:
                    res.append(0)

            print(confusion_matrix(y_test,res))
            logging.info(f"The confusion matrix is {confusion_matrix(y_test,res)}")
            return  accuracy
        except Exception as e:
            raise  custom_exception(e,sys)


    def initiate_model_evaluation(self):
        logging.info("Initiate Model Evaluation")
        try:

            trained_model = keras.models.load_model(self.model_trainer_artifacts.trained_model_path)
            with open("tokenizer.pickle", "rb") as handle:
                tokenizer = pickle.load(handle)
            
            trained_model_accuracy = self.evaluate()

            logging.info("Fetch best model from gcloud storage")
            best_model_path = self.get_best_model_from_gcloud()

            logging.info("Check is the best model present in the gcloud or not ?")

            if os.path.isfile(best_model_path) is False:
                is_model_accepted = True
                logging.info("gcloud storage model is false and currently trained model accepted is True")

            else:
                logging.info("Load best model fetch from gcloud storage")
                best_model = keras.models.load_model(best_model_path)
                best_model_accuracy = self.evaluate(best_model)

                logging.info("Comparing loss between  best model and trained model")
                if best_model_accuracy > trained_model_accuracy:
                    is_model_accepted=True
                    logging.info("Best model is better than trained model")
                else:
                    is_model_accepted=False
                    logging.info("Trained model is better than best model")

            model_evaluation_artifacts = ModelEvaluationArtifacts(is_model_accepted=is_model_accepted)
            logging.info("Returning the ModelEvaluationArtifacts")
            return model_evaluation_artifacts
        
        except Exception  as e:
            raise  custom_exception(e,sys)
