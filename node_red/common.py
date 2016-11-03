from uniflex.core import events

__author__ = "Piotr Gawlowicz"
__copyright__ = "Copyright (c) 2016, Technische Universität Berlin"
__version__ = "0.1.0"
__email__ = "{gawlowicz}@tkn.tu-berlin.de"


class AveragedSpectrumScanSampleEvent(events.EventBase):
    def __init__(self, avg):
        super().__init__()
        self.avg = avg

    def serialize(self):
        return {"avg": self.avg}

    @classmethod
    def parse(cls, buf):
        avg = buf.get("avg", None)
        return cls(avg)


class ChangeWindowSize(events.EventBase):
    def __init__(self, value):
        super().__init__()
        self.window = value


class StartMyFilterEvent(events.EventBase):
    def __init__(self):
        super().__init__()


class StopMyFilterEvent(events.EventBase):
    def __init__(self):
        super().__init__()
