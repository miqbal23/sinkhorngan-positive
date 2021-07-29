import pandas as pd

from base.container import Container


class Histogram(Container):
    def __init__(self, parent, name=None, **kwargs):
        super().__init__(parent=parent, name=name, **kwargs)
        self._data = pd.DataFrame(columns=['arrays_data'])

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def add(self, *new_keys):
        new_keys = pd.DataFrame(index=new_keys, columns=['arrays_data'])
        self._data = self._data.append(new_keys)
        self.reset()

    def updates(self, **kwargs):
        for tag, value in kwargs.items():
            self.update(tag, value)

    def update(self, key, value, n=1):
        if value is None:
            return
        if key not in self._data.index:
            self.add(key)
        self._data.arrays_data[key] = value.clone().cpu().data.numpy()

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

    def flush(self, global_step, run_per: tuple = None):
        for tag, value in self._data.arrays_data.items():
            _tag = "{tag}.___per_{n}_{typ}".format(tag=tag, n=run_per[0], typ=run_per[1]) if run_per else tag
            self.writer.add_histogram(_tag, value, global_step=global_step)
        self.reset()
