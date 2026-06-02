from app.services.s3_service import S3Service


class FakeClient:
    def __init__(self): self.uploaded = []
    def upload_file(self, local_path, bucket, key): self.uploaded.append((local_path, bucket, key))
    def generate_presigned_url(self, *args, **kwargs): return 'url'
    def list_objects_v2(self, **kwargs): return {'Contents': []}


def test_upload_directory_preserves_relative_keys(monkeypatch, tmp_path):
    root = tmp_path / 'root'
    nested = root / 'nested'
    nested.mkdir(parents=True)
    (root / 'a.txt').write_text('a')
    (nested / 'b.txt').write_text('b')
    monkeypatch.setattr('app.services.s3_service.get_settings', lambda: type('S', (), {
        's3_bucket_name':'bucket','s3_default_presigned_url_expiration_in_seconds':3600,'s3_max_presigned_url_expiration_in_seconds':3600,
        's3_region':'','s3_use_path_style':True,'s3_access_key':'','s3_secret_key':'','s3_session_token':'','s3_service_url':''})())
    monkeypatch.setattr('app.services.s3_service.boto3.client', lambda *a, **k: FakeClient())
    svc = S3Service()
    keys = svc.upload_directory(root, 'kaggle-outputs/account/target/job/staging/')
    assert keys == ['kaggle-outputs/account/target/job/staging/a.txt', 'kaggle-outputs/account/target/job/staging/nested/b.txt']


def test_bucket_name_with_prefix_maps_to_bucket_and_key_prefix(monkeypatch, tmp_path):
    (tmp_path / 'a.txt').write_text('a')
    fake = FakeClient()
    monkeypatch.setattr('app.services.s3_service.get_settings', lambda: type('S', (), {
        's3_bucket_name':'shiphard-studio/Dat/Student-Feedback-Sentiment-Analysis','s3_default_presigned_url_expiration_in_seconds':3600,'s3_max_presigned_url_expiration_in_seconds':3600,
        's3_region':'','s3_use_path_style':True,'s3_access_key':'','s3_secret_key':'','s3_session_token':'','s3_service_url':''})())
    monkeypatch.setattr('app.services.s3_service.boto3.client', lambda *a, **k: fake)
    svc = S3Service()
    keys = svc.upload_directory(tmp_path, 'kaggle-outputs/account/target/job/staging/')
    assert svc.bucket == 'shiphard-studio'
    assert keys == ['kaggle-outputs/account/target/job/staging/a.txt']
    assert fake.uploaded[0][1] == 'shiphard-studio'


def test_presign_accepts_logical_key_with_bucket_prefix(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr('app.services.s3_service.get_settings', lambda: type('S', (), {
        's3_bucket_name':'shiphard-studio/Dat/Student-Feedback-Sentiment-Analysis','s3_default_presigned_url_expiration_in_seconds':3600,'s3_max_presigned_url_expiration_in_seconds':3600,
        's3_region':'','s3_use_path_style':True,'s3_access_key':'','s3_secret_key':'','s3_session_token':'','s3_service_url':''})())
    monkeypatch.setattr('app.services.s3_service.boto3.client', lambda *a, **k: fake)
    svc = S3Service()
    svc.presign_get('kaggle-outputs/account/artifact.zip')



def test_upload_directory_rejects_unsafe_prefix(monkeypatch, tmp_path):
    tmp_path.joinpath('a.txt').write_text('a')
    monkeypatch.setattr('app.services.s3_service.get_settings', lambda: type('S', (), {
        's3_bucket_name':'bucket','s3_default_presigned_url_expiration_in_seconds':3600,'s3_max_presigned_url_expiration_in_seconds':3600,
        's3_region':'','s3_use_path_style':True,'s3_access_key':'','s3_secret_key':'','s3_session_token':'','s3_service_url':''})())
    monkeypatch.setattr('app.services.s3_service.boto3.client', lambda *a, **k: FakeClient())
    svc = S3Service()
    try:
        svc.upload_directory(tmp_path, 'bad-prefix/x')
    except ValueError as exc:
        assert 'kaggle-outputs' in str(exc)
    else:
        raise AssertionError('expected ValueError')
