import boto3
import zipfile
import numpy as np
import pandas as pd
import io


class S3Client:
    def __init__(
        self,
        endpoint_url: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        region_name: str,
    ):
        self.client = self._get_client(
            endpoint_url, region_name, aws_access_key_id, aws_secret_access_key
        )

    @staticmethod
    def _get_client(
        endpoint_url, region_name, aws_access_key_id, aws_secret_access_key
    ) -> boto3.client:
        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
        )
        s3 = session.client("s3", endpoint_url=endpoint_url)
        return s3

    def __getattr__(self, attr):
        return getattr(self.client, attr)


def extract_name(f):
    return f.name.rsplit("/", 1)[-1].split(".")[0]


def s3_objects(s3_client, bucket_name, keys, paths):
    if paths is not None:
        for path in paths:
            yield path
    else:
        for key in keys:
            s3_object = s3_client.get_object(Bucket=bucket_name, Key=key)["Body"].read()
            yield io.BytesIO(s3_object)


def read_track_embeddings(s3_client=None, bucket_name=None, keys=None, paths=None):
    track_idx2embeds = {}
    for s3_object in s3_objects(s3_client, bucket_name, keys, paths):
        with zipfile.ZipFile(s3_object) as zf:
            for file in zf.namelist():
                if file.endswith(".npy"):
                    with zf.open(file) as f:
                        track_idx = int(extract_name(f))
                        embeds = np.load(f)
                        track_idx2embeds[track_idx] = embeds
    return track_idx2embeds


def read_tag_data(s3_client=None, bucket_name=None, keys=None, paths=None):
    res = {}
    for s3_object in s3_objects(s3_client, bucket_name, keys, paths):
        with zipfile.ZipFile(s3_object) as zf:
            for file in zf.namelist():
                if file.endswith("train.csv") or file.endswith("test.csv"):
                    with zf.open(file) as f:
                        res[extract_name(f)] = pd.read_csv(f)
    return res
