from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')
    app_name: str = 'Kaggle Notebook Manager'
    env: str = 'development'
    database_url: str = 'postgresql+asyncpg://postgres:postgres@localhost:5432/kaggle_manager'
    admin_api_key: str = 'change-me'
    fernet_key: str = 'generate-fernet-key'
    kaggle_notebook_dir: str = '../notebook/kaggle'
    kaggle_output_dir: str = '../storage/kaggle_outputs'
    kaggle_default_dataset_source: str = 'owner/dataset-slug'
    s3_bucket_name: str = Field('', alias='S3Storage__BucketName')
    s3_region: str = Field('', alias='S3Storage__Region')
    s3_access_key: str = Field('', alias='S3Storage__AccessKey')
    s3_secret_key: str = Field('', alias='S3Storage__SecretKey')
    s3_session_token: str = Field('', alias='S3Storage__SessionToken')
    s3_service_url: str = Field('', alias='S3Storage__ServiceUrl')
    s3_use_path_style: bool = Field(False, alias='S3Storage__UsePathStyle')
    s3_default_presigned_url_expiration_in_seconds: int = Field(3600, alias='S3Storage__DefaultPresignedUrlExpirationInSeconds')
    s3_max_presigned_url_expiration_in_seconds: int = Field(604800, alias='S3Storage__MaxPresignedUrlExpirationInSeconds')

@lru_cache
def get_settings() -> Settings:
    return Settings()
