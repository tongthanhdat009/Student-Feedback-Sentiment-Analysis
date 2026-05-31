import json
from app.services.training_result_parser import parse_training_results


def test_parse_training_results_json(tmp_path):
    (tmp_path / 'training_results.json').write_text(json.dumps({'accuracy': 0.9, 'params': {'epochs': 2}}), encoding='utf-8')
    result = parse_training_results(tmp_path)
    assert result['source_file'] == 'training_results.json'
    assert result['metrics']['accuracy'] == 0.9
    assert result['params']['epochs'] == 2


def test_parse_training_results_txt(tmp_path):
    (tmp_path / 'training_results.txt').write_text('accuracy: 0.91\nepochs=5\nname: demo', encoding='utf-8')
    result = parse_training_results(tmp_path)
    assert result['metrics']['accuracy'] == 0.91
    assert result['params']['epochs'] == 5
    assert result['params']['name'] == 'demo'


def test_parse_training_results_missing_is_warning(tmp_path):
    result = parse_training_results(tmp_path)
    assert result['metrics'] == {}
    assert result['warnings']
