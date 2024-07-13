
from wddetr.utils import LOGGER, SETTINGS, TESTS_RUNNING, colorstr

try:
    # WARNING: do not move import due to protobuf issue in https://github.com/ultralytics/ultralytics/pull/4674
    from torch.utils.tensorboard import SummaryWriter

    assert not TESTS_RUNNING  # do not log pytest
    assert SETTINGS['tensorboard'] is True  # verify integration is enabled
    WRITER = None  # TensorBoard SummaryWriter instance

except (ImportError, AssertionError, TypeError):
    # TypeError for handling 'Descriptors cannot not be created directly.' protobuf errors in Windows
    SummaryWriter = None


def _log_scalars(scalars, step=0):
    if WRITER:
        for k, v in scalars.items():
            WRITER.add_scalar(k, v, step)


def _log_tensorboard_graph(trainer):
    try:
        import warnings

        from ultralytics.utils.torch_utils import de_parallel, torch

        imgsz = trainer.args.imgsz
        imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
        p = next(trainer.model.parameters())  # for device, type
        im = torch.zeros((1, 3, *imgsz), device=p.device, dtype=p.dtype)  # input image (must be zeros, not empty)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)  # suppress jit trace warning
            WRITER.add_graph(torch.jit.trace(de_parallel(trainer.model), im, strict=False), [])
    except Exception as e:
        LOGGER.warning(f'WARNING ⚠️ TensorBoard graph visualization failure {e}')


def on_pretrain_routine_start(trainer):
    if SummaryWriter:
        try:
            global WRITER
            WRITER = SummaryWriter(str(trainer.save_dir))
            prefix = colorstr('TensorBoard: ')
            LOGGER.info(f"{prefix}Start with 'tensorboard --logdir {trainer.save_dir}', view at http://localhost:6006/")
        except Exception as e:
            LOGGER.warning(f'WARNING ⚠️ TensorBoard not initialized correctly, not logging this run. {e}')


def on_train_start(trainer):
    if WRITER:
        _log_tensorboard_graph(trainer)


def on_batch_end(trainer):
    _log_scalars(trainer.label_loss_items(trainer.tloss, prefix='train'), trainer.epoch + 1)


def on_fit_epoch_end(trainer):
    _log_scalars(trainer.metrics, trainer.epoch + 1)


callbacks = {
    'on_pretrain_routine_start': on_pretrain_routine_start,
    'on_train_start': on_train_start,
    'on_fit_epoch_end': on_fit_epoch_end,
    'on_batch_end': on_batch_end} if SummaryWriter else {}
