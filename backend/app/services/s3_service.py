from datetime import datetime, timedelta, timezone
from pathlib import Path
from botocore.config import Config
from botocore.exceptions import ClientError
import boto3
from ..config import get_settings
from ..utils.path_guard import normalize_s3_key

class S3Service:
    def __init__(self):
        s = get_settings()
        if not s.s3_bucket_name: raise RuntimeError('S3Storage__BucketName is required')
        bucket, _, prefix = s.s3_bucket_name.strip('/').partition('/')
        self.bucket = bucket
        self.key_prefix = prefix.strip('/')
        self.default_exp = s.s3_default_presigned_url_expiration_in_seconds
        self.max_exp = min(s.s3_max_presigned_url_expiration_in_seconds, 604800)
        kwargs = {'region_name': s.s3_region or None, 'config': Config(signature_version='s3v4', s3={'addressing_style':'path' if s.s3_use_path_style else 'auto'}, retries={'max_attempts':3,'mode':'adaptive'})}
        if s.s3_access_key: kwargs['aws_access_key_id'] = s.s3_access_key
        if s.s3_secret_key: kwargs['aws_secret_access_key'] = s.s3_secret_key
        if s.s3_session_token: kwargs['aws_session_token'] = s.s3_session_token
        if s.s3_service_url: kwargs['endpoint_url'] = s.s3_service_url
        self.client = boto3.client('s3', **kwargs)
    def _object_key(self, logical_key: str) -> str:
        key = normalize_s3_key(logical_key)
        return f'{self.key_prefix}/{key}' if self.key_prefix else key
    def upload_file(self, local_path: str | Path, object_key: str) -> str:
        key = self._object_key(object_key)
        try: self.client.upload_file(str(local_path), self.bucket, key)
        except ClientError as exc: raise RuntimeError(str(exc)) from exc
        return key
    def upload_directory(self, local_dir: str | Path, prefix: str) -> list[str]:
        root = Path(local_dir)
        if not root.is_dir(): raise ValueError('local_dir must be a directory')
        clean_prefix = prefix.rstrip('/') + '/'
        uploaded: list[str] = []
        for path in sorted(p for p in root.rglob('*') if p.is_file()):
            rel = path.relative_to(root).as_posix()
            uploaded.append(self.upload_file(path, clean_prefix + rel))
        return uploaded
    def presign_get(self, object_key: str, expires_in: int | None = None):
        key = self._object_key(object_key); exp = min(expires_in or self.default_exp, self.max_exp)
        url = self.client.generate_presigned_url('get_object', Params={'Bucket': self.bucket, 'Key': key}, ExpiresIn=exp)
        return url, datetime.now(timezone.utc) + timedelta(seconds=exp)
    def list_prefix(self, prefix: str):
        prefix = self._object_key(prefix)
        return self.client.list_objects_v2(Bucket=self.bucket, Prefix=prefix).get('Contents', [])
