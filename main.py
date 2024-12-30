from libs.app import Application


SERVER_IP = "192.168.1.26"
API_BASE_URL = f"http://{SERVER_IP}:8003"

MINIO_ACCESS_KEY = "BVBJYWOU8gaL9sytgpH0"
MINIO_SECRET_KEY = "YWdP3s3rhZoI6RrjLpOkoP53mf5D56bz0S7YXSnT"
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
        threshold=0.5,
        top_n=10,
        max_workers=4
    )
    application.run(offset="latest")


if __name__ == "__main__":
    main()