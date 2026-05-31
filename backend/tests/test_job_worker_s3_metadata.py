from pathlib import Path
from app.services.job_worker import choose_primary_artifact, job_s3_prefix


def test_job_s3_prefix_stays_under_kaggle_outputs():
    assert job_s3_prefix('acc', 'target', 'job') == 'kaggle-outputs/acc/target/job'


def test_choose_primary_artifact_prefers_manifest_pattern(tmp_path):
    out = tmp_path
    (out / 'model.pt').write_text('x')
    (out / 'metrics.json').write_text('{}')
    assert choose_primary_artifact(out, ['metrics.json']).name == 'metrics.json'


def test_choose_primary_artifact_falls_back_priority(tmp_path):
    (tmp_path / 'a.csv').write_text('x')
    (tmp_path / 'model.pt').write_text('x')
    assert choose_primary_artifact(Path(tmp_path)).name == 'model.pt'
