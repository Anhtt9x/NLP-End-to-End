from dataclasses import dataclass
import os

@dataclass
class DataIngestionArtifacts:
    imbalance_data_file_path:str
    raw_data_file_path:str

    
@dataclass
class DataTranformationArtifacts:
    transformed_data_path:str


@dataclass
class ModelTrainerArtifacts:
    trained_model_path:str
    x_test_path:list
    y_test_path:list