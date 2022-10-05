from .api import ScreenVAE_rec as SVAE

class _highgui_containerclass:
    def __init__(self):
        from .visual import main as _decoder_preview
        self.decoder_preview = _decoder_preview
        from .edit import main as _process_image
        self.process_image = _process_image

highgui = _highgui_containerclass()

__all__ = ['SVAE', 'highgui']