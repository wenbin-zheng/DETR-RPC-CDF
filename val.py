import warnings
warnings.filterwarnings('ignore')
from wddetr import RTDETR

if __name__ == '__main__':
    model = RTDETR('weed-detr.pt')
    model.val(data='data.yaml',
              split='test',
              imgsz=640,
              batch=4,
              project='runs/val',
              name='exp',
              )