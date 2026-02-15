

import os
import sys
sys.path.append(os.getcwd())
import gdown
import zipfile
from src.logger import get_logger
from src.custom_exception import CustomException
from config.data_ingestion_config import GDRIVE_FILE_ID, TARGET_DIR

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self):
        self.file_id = GDRIVE_FILE_ID
        self.target_dir = TARGET_DIR

    def download_and_extract(self):
        try:
      
            raw_dir = os.path.join(self.target_dir, "data_ingestion")
            os.makedirs(raw_dir, exist_ok=True)
            
            zip_path = os.path.join(raw_dir, "visdrone_data.zip")
            
         
            if not os.path.exists(zip_path):
                logger.info(f"Downloading VisDrone subset from GDrive ID: {self.file_id}")
                url = f'https://drive.google.com/uc?id={self.file_id}'
                gdown.download(url, zip_path, quiet=False)
            else:
                logger.info("visdrone_data.zip already exists. Skipping download.")

          
            extract_path = os.path.join(raw_dir, "VisDrone_Dataset")
            if not os.path.exists(extract_path):
                logger.info(f"Extracting to {extract_path}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
                logger.info("Extraction Complete.")
            else:
                logger.info("Directory already exists. Skipping extraction.")

            return extract_path

        except Exception as e:
            raise CustomException("Data Ingestion Stage Failed", e)

if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.download_and_extract()