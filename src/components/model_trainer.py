import os
import sys
import pickle
import pandas as pd
from src.logger.logging import logging
from src.exception.exception import custom_exception
from src.constants import *
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from src.entity.artifact_entity import ModelTrainerArtifacts,DataTranformationArtifacts
from src.entity.config_entity import ModelTrainerConfig
from src.model.base_model import ModelArchitecture

class ModelTrainer:
    def __init__(self, data_transformation_artifacts: DataTranformationArtifacts,
                 model_trainer_config: ModelTrainerConfig):
        
        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_trainer_config = model_trainer_config

    def spliting_data(self, csv_path):
        try:
            logging.info("Entered the spliting data function")
            logging.info("Reading the data")
            df = pd.read_csv(csv_path,index_col=False)

            logging.info("Spliting the data into x and y")
            x = df[TWEET]
            y = df[LABEL]

            logging.info("Applying train test split on the data")
            x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3, random_state=42)
            logging.info("Exited the spliting the data function")
            return x_train,x_test,y_train,y_test
        
        except Exception as e:
            raise custom_exception(e,sys)
        
    def tokenizing(self, x_train):
        try:
            logging.info("Entered the tokenizing function")
            tokenizer = Tokenizer(num_words=self.model_trainer_config.MAX_WORDS)
            tokenizer.fit_on_texts(x_train)
            sequences = tokenizer.texts_to_sequences(x_train)

            logging.info(f"Converting text to sequences: {sequences}")

            sequences_matrix = pad_sequences(sequences,maxlen=self.model_trainer_config.MAX_LEN)
            logging.info(f"Converting sequences to matrix: {sequences_matrix}")

            return  sequences_matrix,tokenizer
        
        except Exception as e:
            raise custom_exception(e,sys)


    def initiate_model_trainer(self)  -> ModelTrainerArtifacts:
        try:
            logging.info("Entered the initiate model trainer function")
            x_train,  x_test, y_train, y_test = self.spliting_data(self.data_transformation_artifacts.transformed_data_path)
            model_architecture = ModelArchitecture()

            model = model_architecture.create_model()
            logging.info("Compiling the model")

            sequences_matrix, tokenizer = self.tokenizing(x_train)

            logging.info("Entered into the model training")

            model.fit(sequences_matrix,y_train,
                      batch_size=self.model_trainer_config.BATCH_SIZE,
                      epochs=self.model_trainer_config.EPOCH,
                      validation_split=self.model_trainer_config.VALIDATION_SPLIT)
            
            logging.info("Model training finished")
            with open("tokenizer.pickle", "wb") as handle:
                pickle.dump(tokenizer,  handle, protocol=pickle.HIGHEST_PROTOCOL)
                logging.info("Model artifacts saved")
            os.makedirs(self.model_trainer_config.TRAINED_MODEL_DIR,exist_ok=True)

            logging.info("Saving the model")
            model.save(self.model_trainer_config.TRAINED_MODEL_PATH)
            logging.info("Model saved")
            
            x_test.to_csv(self.model_trainer_config.X_TEST_DATA_PATH)
            y_test.to_csv(self.model_trainer_config.Y_TEST_DATA_PATH)
            x_train.to_csv(self.model_trainer_config.X_TRAIN_DATA_PATH)

            model_trainer_artifacts = ModelTrainerArtifacts(
                model_path=self.model_trainer_config.TRAINED_MODEL_PATH,
                x_test_path=self.model_trainer_config.X_TEST_DATA_PATH,
                y_test_path=self.model_trainer_config.Y_TEST_DATA_PATH,
            )

            logging.info("Return the ModelTrainerArtifacts")
            return model_trainer_artifacts
        
        except Exception as e:
            raise  custom_exception(e,sys)
        
        