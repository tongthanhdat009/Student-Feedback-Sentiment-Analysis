from app.services.job_worker import get_kernel_status, normalize_kaggle_status


class LegacyApi:
    def __init__(self): self.called = None
    def kernels_status(self, ref):
        self.called = ref
        return {'status': 'complete'}


class NewApi:
    def __init__(self): self.called = None
    def kernel_status(self, ref):
        self.called = ref
        return {'status': 'running'}


def test_describe_kaggle_status_expands_empty_failure_message():
    from app.services.job_worker import describe_kaggle_status
    msg = describe_kaggle_status({'status': 'ERROR', 'failureMessage': ''})
    assert 'failed without a failureMessage' in msg
    assert 'Kaggle kernel run log' in msg


def test_describe_kaggle_status_includes_failure_message():
    from app.services.job_worker import describe_kaggle_status
    assert describe_kaggle_status({'status': 'ERROR', 'failureMessage': 'boom'}) == 'Kaggle ERROR: boom'


def test_normalize_kaggle_status():
    assert normalize_kaggle_status('complete') == 'completed'
    assert normalize_kaggle_status({'status': 'succeeded'}) == 'completed'
    assert normalize_kaggle_status('running') == 'running'
    assert normalize_kaggle_status('cancelled') == 'failed'
    assert normalize_kaggle_status('error') == 'failed'


def test_get_kernel_status_supports_installed_kaggle_sdk_method():
    api = LegacyApi()
    assert get_kernel_status(api, 'u/s') == {'status': 'complete'}
    assert api.called == 'u/s'


def test_get_kernel_status_prefers_newer_singular_method_if_present():
    api = NewApi()
    assert get_kernel_status(api, 'u/s') == {'status': 'running'}
    assert api.called == 'u/s'


def test_failed_statuses_not_download_ready():
    not_ready = {'pending', 'staging', 'pushed', 'running', 'failed'}
    assert 'completed' not in not_ready


def test_worker_code_does_not_lazy_load_job_account_relationship():
    from pathlib import Path
    source = Path('app/services/job_worker.py').read_text(encoding='utf-8')
    assert 'job.account or' not in source
    assert 'acc=job.account' not in source


def test_format_job_error_explains_kaggle_push_401():
    from app.services.job_worker import format_job_error
    msg = format_job_error(Exception('401 Client Error: Unauthorized for url: https://api.kaggle.com/v1/kernels.KernelsApiService/SaveKernel'))
    assert 'Kaggle push unauthorized' in msg
    assert 'API key' in msg


def test_worker_uploads_staging_then_cleans_local_folder():
    from pathlib import Path
    source = Path('app/services/job_worker.py').read_text(encoding='utf-8')
    assert 'uploaded_staging' in source
    assert 'no files were uploaded' in source
    assert 'shutil.rmtree' in source
    assert 'job.staging_path = None' in source


def test_worker_poll_treats_kaggle_status_404_as_not_ready():
    from pathlib import Path
    source = Path('app/services/job_worker.py').read_text(encoding='utf-8')
    assert 'status_code' in source
    assert 'Kaggle status not ready yet (404)' in source
    assert "yield 'running', status" in source


def test_worker_bounds_kaggle_status_404_wait():
    from pathlib import Path
    source = Path('app/services/job_worker.py').read_text(encoding='utf-8')
    assert 'status_not_found_grace_seconds' in source
    assert 'stayed 404' in source
