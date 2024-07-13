

import requests

from wddetr.data.utils import HUBDatasetStats
from wddetr.hub.auth import Auth
from wddetr.hub.utils import HUB_API_ROOT, HUB_WEB_ROOT, PREFIX
from wddetr.utils import LOGGER, SETTINGS


def login(api_key=''):

    Auth(api_key, verbose=True)


def logout():

    SETTINGS['api_key'] = ''
    SETTINGS.save()
    LOGGER.info(f"{PREFIX}logged out ✅. To log in again, use 'yolo hub login'.")


def reset_model(model_id=''):

    r = requests.post(f'{HUB_API_ROOT}/model-reset', json={'apiKey': Auth().api_key, 'modelId': model_id})
    if r.status_code == 200:
        LOGGER.info(f'{PREFIX}Model reset successfully')
        return
    LOGGER.warning(f'{PREFIX}Model reset failure {r.status_code} {r.reason}')


def export_fmts_hub():

    from ultralytics.engine.exporter import export_formats
    return list(export_formats()['Argument'][1:]) + ['ultralytics_tflite', 'ultralytics_coreml']


def export_model(model_id='', format='torchscript'):

    assert format in export_fmts_hub(), f"Unsupported export format '{format}', valid formats are {export_fmts_hub()}"
    r = requests.post(f'{HUB_API_ROOT}/v1/models/{model_id}/export',
                      json={'format': format},
                      headers={'x-api-key': Auth().api_key})
    assert r.status_code == 200, f'{PREFIX}{format} export failure {r.status_code} {r.reason}'
    LOGGER.info(f'{PREFIX}{format} export started ✅')


def get_export(model_id='', format='torchscript'):
    """Get an exported model dictionary with download URL."""
    assert format in export_fmts_hub(), f"Unsupported export format '{format}', valid formats are {export_fmts_hub()}"
    r = requests.post(f'{HUB_API_ROOT}/get-export',
                      json={
                          'apiKey': Auth().api_key,
                          'modelId': model_id,
                          'format': format})
    assert r.status_code == 200, f'{PREFIX}{format} get_export failure {r.status_code} {r.reason}'
    return r.json()


def check_dataset(path='', task='detect'):

    HUBDatasetStats(path=path, task=task).get_json()
    LOGGER.info(f'Checks completed correctly ✅. Upload this dataset to {HUB_WEB_ROOT}/datasets/.')
