
import subprocess

from wddetr.cfg import TASK2DATA, TASK2METRIC, get_save_dir
from wddetr.utils import DEFAULT_CFG, DEFAULT_CFG_DICT, LOGGER, NUM_THREADS


def run_ray_tune(model,
                 space: dict = None,
                 grace_period: int = 10,
                 gpu_per_trial: int = None,
                 max_samples: int = 10,
                 **train_args):


    LOGGER.info('💡 Learn about RayTune at https://docs.ultralytics.com/integrations/ray-tune')
    if train_args is None:
        train_args = {}

    try:
        subprocess.run('pip install ray[tune]'.split(), check=True)

        import ray
        from ray import tune
        from ray.air import RunConfig
        from ray.air.integrations.wandb import WandbLoggerCallback
        from ray.tune.schedulers import ASHAScheduler
    except ImportError:
        raise ModuleNotFoundError('Tuning hyperparameters requires Ray Tune. Install with: pip install "ray[tune]"')

    try:
        import wandb

        assert hasattr(wandb, '__version__')
    except (ImportError, AssertionError):
        wandb = False

    default_space = {
        # 'optimizer': tune.choice(['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp']),
        'lr0': tune.uniform(1e-5, 1e-1),
        'lrf': tune.uniform(0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
        'momentum': tune.uniform(0.6, 0.98),  # SGD momentum/Adam beta1
        'weight_decay': tune.uniform(0.0, 0.001),  # optimizer weight decay 5e-4
        'warmup_epochs': tune.uniform(0.0, 5.0),  # warmup epochs (fractions ok)
        'warmup_momentum': tune.uniform(0.0, 0.95),  # warmup initial momentum
        'box': tune.uniform(0.02, 0.2),  # box loss gain
        'cls': tune.uniform(0.2, 4.0),  # cls loss gain (scale with pixels)
        'hsv_h': tune.uniform(0.0, 0.1),  # image HSV-Hue augmentation (fraction)
        'hsv_s': tune.uniform(0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
        'hsv_v': tune.uniform(0.0, 0.9),  # image HSV-Value augmentation (fraction)
        'degrees': tune.uniform(0.0, 45.0),  # image rotation (+/- deg)
        'translate': tune.uniform(0.0, 0.9),  # image translation (+/- fraction)
        'scale': tune.uniform(0.0, 0.9),  # image scale (+/- gain)
        'shear': tune.uniform(0.0, 10.0),  # image shear (+/- deg)
        'perspective': tune.uniform(0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
        'flipud': tune.uniform(0.0, 1.0),  # image flip up-down (probability)
        'fliplr': tune.uniform(0.0, 1.0),  # image flip left-right (probability)
        'mosaic': tune.uniform(0.0, 1.0),  # image mixup (probability)
        'mixup': tune.uniform(0.0, 1.0),  # image mixup (probability)
        'copy_paste': tune.uniform(0.0, 1.0)}  # segment copy-paste (probability)

    # Put the model in ray store
    task = model.task
    model_in_store = ray.put(model)

    def _tune(config):
        model_to_train = ray.get(model_in_store)  # get the model from ray store for tuning
        model_to_train.reset_callbacks()
        config.update(train_args)
        results = model_to_train.train(**config)
        return results.results_dict

    if not space:
        space = default_space
        LOGGER.warning('WARNING ⚠️ search space not provided, using default search space.')

    data = train_args.get('data', TASK2DATA[task])
    space['data'] = data
    if 'data' not in train_args:
        LOGGER.warning(f'WARNING ⚠️ data not provided, using default "data={data}".')

    trainable_with_resources = tune.with_resources(_tune, {'cpu': NUM_THREADS, 'gpu': gpu_per_trial or 0})

    asha_scheduler = ASHAScheduler(time_attr='epoch',
                                   metric=TASK2METRIC[task],
                                   mode='max',
                                   max_t=train_args.get('epochs') or DEFAULT_CFG_DICT['epochs'] or 100,
                                   grace_period=grace_period,
                                   reduction_factor=3)

    tuner_callbacks = [WandbLoggerCallback(project='YOLOv8-tune')] if wandb else []

    tune_dir = get_save_dir(DEFAULT_CFG, name='tune').resolve()  # must be absolute dir
    tune_dir.mkdir(parents=True, exist_ok=True)
    tuner = tune.Tuner(trainable_with_resources,
                       param_space=space,
                       tune_config=tune.TuneConfig(scheduler=asha_scheduler, num_samples=max_samples),
                       run_config=RunConfig(callbacks=tuner_callbacks, storage_path=tune_dir))

    tuner.fit()

    return tuner.get_results()
