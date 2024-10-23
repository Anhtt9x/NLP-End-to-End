from dataclasses import dataclass
import os

@dataclass
class DataIngestionArtifacts:
    imbalance_data_file_path:str
    raw_data_file_path:str

    