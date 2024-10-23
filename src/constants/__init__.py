import os
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
ARTIFACT_DIR = os.path.join("artifacts",TIMESTAMP)
BUCKET_NAME=""
ZIP_FILE_NAME="dataset.zip"
LABEL="label"
TWEET="tweet"

DATA_INGESTION_ARTIFACTS_DIR = "DataIngestionArtifacts"
DATA_INGESTION_IMBALANCE_DATA_DIR = "imbalance_data.csv"
DATA_INGESTION_RAW_DATA_DIR =  "raw_data.csv"