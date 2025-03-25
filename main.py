from libs.app import Application

import os
from dotenv import load_dotenv

load_dotenv()  

SERVER_IP = os.getenv("IP_ADDRESS")

BACKEND_ADDRESS = "http://"+SERVER_IP+":3001"

BACKEND_EMAIL = os.getenv("BACKEND_EMAIL", "admin@example.com")
BACKEND_PASSWORD = os.getenv("BACKEND_PASSWORD", "Adminpassword1@")

API_BASE_URL = f"http://{SERVER_IP}:8003"

MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_URL = f"{SERVER_IP}:9000"

brokers = [f'{SERVER_IP}:9092']

BUCKET_NAME = "my-bucket"

TOPIC_INPUT = "similarity-calculation"
TOPIC_OUTPUT = "similarity-calculation-results"
WORKING_FOLDER = "./tmp"
CROPPED_IMAGE_PREFIX = "croped_yolo"

def main():

    application = Application(
        brokers=brokers,
        server_ip=SERVER_IP,
        minio_access_key=MINIO_ACCESS_KEY,
        minio_secret_key=MINIO_SECRET_KEY,
        working_dir=WORKING_FOLDER,
        topic_input=TOPIC_INPUT,
        topic_output=TOPIC_OUTPUT,
        bucket_name=BUCKET_NAME,
        croped_image_prefix=CROPPED_IMAGE_PREFIX,
        backend_address=BACKEND_ADDRESS,
        email=BACKEND_EMAIL,
        password=BACKEND_PASSWORD,
        threshold=0.5,
        top_n=10,
        max_workers=4
    )
    application.run(offset="latest")


if __name__ == "__main__":
    main()