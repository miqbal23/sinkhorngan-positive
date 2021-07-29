import modules
from base.container import Container


class DataLoader(Container):
    def __init__(self, parent, **kwargs):
        super().__init__(parent=parent, name='dataloader', **kwargs)
        self.set_modules(module=modules.dataloader)
