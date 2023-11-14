import os
from dotenv import load_dotenv

load_dotenv()

S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
S3_AWS_ACCESS_KEY_ID = os.getenv("S3_AWS_ACCESS_KEY_ID")
S3_AWS_SECRET_ACCESS_KEY = os.getenv("S3_AWS_SECRET_ACCESS_KEY")
S3_REGION_NAME = os.getenv("S3_REGION_NAME")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_KEY_BASE = os.getenv("S3_KEY_BASE")
