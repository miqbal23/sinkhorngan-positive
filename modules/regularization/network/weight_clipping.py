from base.container import Container


class WeightClipping(Container):

    def __call__(self, network, **kwargs):

        c = self.configs['clip']

        if c is None:
            return

        if isinstance(c, float) or isinstance(c, int):
            c = (c * -1, c)
        else:
            if len(c) != 2:
                raise ValueError('Please only give 2 clipping input')

        for p in network.parameters():
            p.data.clamp_(c[0], c[1])
