
__version__ = '8.0.201'

from models import WeedDETR


from utils import SETTINGS as settings
from utils.checks import check_yolo as checks
from utils.downloads import download

__all__ = '__version__',  'WeedDETR', 'checks', 'download', 'settings'
