
from wddetr.engine.model import Model
from wddetr.nn.tasks import RTDETRDetectionModel

from .predict import RTDETRPredictor
from .train import RTDETRTrainer
from .val import RTDETRValidator


class RTDETR(Model):


    def __init__(self, model='rtdetr-l.pt') -> None:

        if model and model.split('.')[-1] not in ('pt', 'yaml', 'yml'):
            raise NotImplementedError('RT-DETR only supports creating from *.pt, *.yaml, or *.yml files.')
        super().__init__(model=model, task='detect')

    @property
    def task_map(self) -> dict:

        return {
            'detect': {
                'predictor': RTDETRPredictor,
                'validator': RTDETRValidator,
                'trainer': RTDETRTrainer,
                'model': RTDETRDetectionModel}}
