import boto3
import zipfile
import numpy as np
import pandas as pd
import io
import os


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


def _extract_name(f):
    return f.name.rsplit("/", 1)[-1].split(".")[0]


def extract_name(f):
    return f.rsplit("/", 1)[-1].split(".")[0]


def s3_objects(s3_client, bucket_name, keys, paths):
    if paths is not None:
        for path in paths:
            yield path
    else:
        for key in keys:
            s3_object = s3_client.get_object(Bucket=bucket_name, Key=key)["Body"].read()
            yield io.BytesIO(s3_object)


def _load_track_embeddings(s3_client=None, bucket_name=None, keys=None, paths=None):
    track_idx2embeds = {}
    for s3_object in s3_objects(s3_client, bucket_name, keys, paths):
        with zipfile.ZipFile(s3_object) as zf:
            for file in zf.namelist():
                if file.endswith(".npy"):
                    with zf.open(file) as f:
                        track_idx = int(_extract_name(f))
                        embeds = np.load(f)
                        track_idx2embeds[track_idx] = embeds
    return track_idx2embeds


def _load_tag_data(s3_client=None, bucket_name=None, keys=None, paths=None):
    res = {}
    for s3_object in s3_objects(s3_client, bucket_name, keys, paths):
        with zipfile.ZipFile(s3_object) as zf:
            for file in zf.namelist():
                if file.endswith("train.csv") or file.endswith("test.csv"):
                    with zf.open(file) as f:
                        res[_extract_name(f)] = pd.read_csv(f)
    return res


def _load_track_knn(s3_client=None, bucket_name=None, keys=None, paths=None):
    track_idx2knn = {}
    for s3_object in s3_objects(s3_client, bucket_name, keys, paths):
        with zipfile.ZipFile(s3_object) as zf:
            for file in zf.namelist():
                if file.endswith(".npy"):
                    with zf.open(file) as f:
                        track_idx = int(_extract_name(f))
                        embeds = np.load(f)
                        track_idx2knn[track_idx] = embeds
    return track_idx2knn


def _load_data(cfg):
    tag_data = _load_tag_data(paths=[os.path.join(cfg["data_path"], "data.zip")])
    track_idx2embeds = _load_track_embeddings(
        paths=[
            os.path.join(cfg["data_path"], "track_embeddings", f"dir_00{i}.zip")
            for i in range(1, 9)
        ],
    )
    track_idx2knn = None
    if cfg.get("knn_data", False):
        track_idx2knn = _load_track_knn(
            paths=[os.path.join(cfg["data_path"], "knn_data.zip")],
        )
    return tag_data, track_idx2embeds, track_idx2knn


def load_track_embeddings(s3_client=None, bucket_name=None, keys=None, paths=None):
    track_idx2embeds = {}
    path = paths[0]
    for file in os.listdir(path):
        if file.endswith(".npy"):
            track_idx = int(extract_name(file))
            embeds = np.load(os.path.join(path, file))
            track_idx2embeds[track_idx] = embeds
    return track_idx2embeds


def load_tag_data(s3_client=None, bucket_name=None, keys=None, paths=None):
    res = {}
    path = paths[0]
    for file in os.listdir(path):
        if file.endswith("train.csv") or file.endswith("test.csv"):
            res[extract_name(file)] = pd.read_csv(os.path.join(path, file))
    return res


# def load_track_knn(s3_client=None, bucket_name=None, keys=None, paths=None):
#     track_idx2knn = {}
#     for s3_object in s3_objects(s3_client, bucket_name, keys, paths):
#         with zipfile.ZipFile(s3_object) as zf:
#             for file in zf.namelist():
#                 if file.endswith(".npy"):
#                     with zf.open(file) as f:
#                         track_idx = int(extract_name(f))
#                         embeds = np.load(f)
#                         track_idx2knn[track_idx] = embeds
#     return track_idx2knn


def load_data(cfg):
    tag_data = load_tag_data(paths=[os.path.join(cfg["data_path"], "data")])
    track_idx2embeds = load_track_embeddings(
        paths=[os.path.join(cfg["data_path"], "track_embeddings")]
    )
    track_idx2knn = None
    if cfg.get("knn_data", False):
        raise NotImplementedError("knn_data in demo mode is not supported")
    return tag_data, track_idx2embeds, track_idx2knn
