

from wddetr.utils import emojis


class HUBModelError(Exception):


    def __init__(self, message='Model not found. Please check model URL and try again.'):
        super().__init__(emojis(message))
