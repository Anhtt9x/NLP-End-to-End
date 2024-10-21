import logging
from datetime import datetime
import os
from from_root import from_root


LOG_PATH = f"{datetime.now().strftime("%m-%d-%y-%H-%M-%S")}.log"

FILE_LOG = os.path.join(os.getcwd(),"logs")

os.makedirs(FILE_LOG,exist_ok=True)

LOG_FILE_PATH = os.path.join(FILE_LOG,LOG_PATH)

logging.basicConfig(filename=LOG_FILE_PATH,
                    level=logging.INFO,
                    format="[%(asctime)s] %(lineno)d %(name)s-%(levelname)s-%(message)s")


