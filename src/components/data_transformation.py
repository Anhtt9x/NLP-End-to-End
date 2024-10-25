import os
import re
import sys
from src.exception.exception import custom_exception
from src.logger.logging import logging
import string
import pandas as pd
import nltk
from nltk.corpus import  stopwords
nltk.download('stopwords')
from sklearn.model_selection import train_test_split
from src.entity.config_entity import DataTranformationConfig
from src.entity.artifact_entity import DataIngestionArtifacts,DataTranformationArtifacts

class DataTranformation:
    def __init__(self, data_transformation_config: DataTranformationConfig, 
                 data_ingestion_artifacts: DataIngestionArtifacts):

        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifacts = data_ingestion_artifacts

    
    def imbalance_data_cleaning(self):
        try:
            logging.info("Entered into the imbalance data cleaning class")
            imbalance_data = pd.read_csv(self.data_ingestion_artifacts.imbalance_data_file_path)

            logging.info("Imbalance data loaded successfully")
            imbalance_data.drop(self.data_transformation_config.ID,
                                axis=self.data_transformation_config.AXIS,
                                inplace=self.data_transformation_config.INPLACE)
            logging.info("Exited the imbalance data_cleaning function")

            return  imbalance_data
        
        except Exception as e:
            raise custom_exception(e, sys)

    def raw_data_cleaning(self):
        try:
            logging.info("Entered into the  raw data cleaning class")
            raw_data = pd.read_csv(self.data_ingestion_artifacts.raw_data_file_path)

            logging.info("Raw data loaded successfully")
            raw_data.drop(self.data_transformation_config.DROP_COLUMNS
                          ,axis=self.data_transformation_config.AXIS,
                          inplace=self.data_transformation_config.INPLACE)
            
            raw_data.loc[raw_data[self.data_transformation_config.CLASS] == 0, self.data_transformation_config.CLASS] = 1

            raw_data[self.data_transformation_config.CLASS].replace({0:1},inplace=True)
            raw_data[self.data_transformation_config.CLASS].replace({2:0}, inplace=True)

            raw_data.rename(columns={self.data_transformation_config.CLASS:self.data_transformation_config.LABEL},
                            inplace=True)
            logging.info(f"Entered the raw data cleaning function and return raw data {raw_data}")

            return raw_data
        
        except Exception as e:
            raise custom_exception(e,sys)
            
    def concat_the_frame(self):
        try:
            logging.info("Entered into the concat frame function")
            frame = [self.imbalance_data_cleaning(), self.raw_data_cleaning()]

            df = pd.concat(frame)
            print(df.head())

            logging.info(f"Return the concat dataframe {df}")

            return df
        
        except Exception as e:
            raise  custom_exception(e,sys)

    def concat_data_cleaning(self, words):

        try:
            logging.info("Entered into the concat_data_cleaning function")
            # Let's apply stemming and stopwords on the data
            stemmer = nltk.SnowballStemmer("english")
            stopword = set(stopwords.words('english'))
            words = str(words).lower()
            words = re.sub('\[.*?\]', '', words)
            words = re.sub('https?://\S+|www\.\S+', '', words)
            words = re.sub('<.*?>+', '', words)
            words = re.sub('[%s]' % re.escape(string.punctuation), '', words)
            words = re.sub('\n', '', words)
            words = re.sub('\w*\d\w*', '', words)
            words = [word for word in words.split(' ') if words not in stopword]
            words=" ".join(words)
            words = [stemmer.stem(word) for word in words.split(' ')]
            words=" ".join(words)
            logging.info("Exited the concat_data_cleaning function")
            return words 

        except Exception as e:
            raise custom_exception(e,sys)
        

    def initiate_data_transform(self):
        try:
            logging.info("Entered the  initiate_data_transform function")
            self.imbalance_data_cleaning()
            self.raw_data_cleaning()
            df=self.concat_the_frame()
            df[self.data_transformation_config.TWEET] = df[self.data_transformation_config.TWEET].apply(self.concat_data_cleaning)

            os.makedirs(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR,
                        exist_ok=True)
            
            df.to_csv(self.data_transformation_config.TRANSFORMED_FILE_NAME,
                      index=False,header=True)
            
            data_transformation_artifacts = DataTranformationArtifacts(
                self.data_transformation_config.TRANSFORMED_FILE_NAME
            )

            logging.info("Return TransformationArtifacts")
            return data_transformation_artifacts
        
        except Exception as e:
            raise custom_exception(e,sys)
