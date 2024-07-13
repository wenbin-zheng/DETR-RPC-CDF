
from wddetr.utils import SETTINGS

try:
    assert SETTINGS['raytune'] is True  # verify integration is enabled
    import ray
    from ray import tune
    from ray.air import session

except (ImportError, AssertionError):
    tune = None


def on_fit_epoch_end(trainer):
    if ray.tune.is_session_enabled():
        metrics = trainer.metrics
        metrics['epoch'] = trainer.epoch
        session.report(metrics)


callbacks = {
    'on_fit_epoch_end': on_fit_epoch_end, } if tune else {}
