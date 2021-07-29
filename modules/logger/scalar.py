import pandas as pd

from base.container import Container


class Scalar(Container):
    def __init__(self, parent, name=None, **kwargs):
        super().__init__(parent=parent, name=name, **kwargs)
        self._data = pd.DataFrame(columns=['total', 'counts', 'average'])

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def add(self, *new_keys):
        new_keys = pd.DataFrame(index=new_keys, columns=['total', 'counts', 'average'])
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
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

    def flush(self, global_step, run_per: tuple = None):
        for tag, value in self._data.average.items():
            _tag = "{tag}.___per_{n}_{typ}".format(tag=tag, n=run_per[0], typ=run_per[1]) if run_per else tag
            self.writer.add_scalar(_tag, value, global_step=global_step)
        self.reset()
