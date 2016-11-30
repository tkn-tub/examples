from uniflex.core import events

__author__ = "Piotr Gawlowicz"
__copyright__ = "Copyright (c) 2016, Technische Universität Berlin"
__version__ = "0.1.0"
__email__ = "{gawlowicz}@tkn.tu-berlin.de"


class StaStateEvent(events.EventBase):
    def __init__(self, sta, state):
        super().__init__()
        self.sta = sta
        self.state = state

    def serialize(self):
        return {"sta": self.sta, "state": self.state}

    @classmethod
    def parse(cls, buf):
        sta = buf.get("sta", None)
        state = buf.get("state", None)
        return cls(sta, state)


class StaThroughputEvent(events.EventBase):
    def __init__(self, sta, throughput):
        super().__init__()
        self.sta = sta
        self.throughput = throughput

    def serialize(self):
        return {"sta": self.sta, "throughput": self.throughput}

    @classmethod
    def parse(cls, buf):
        sta = buf.get("sta", None)
        throughput = buf.get("throughput", None)
        return cls(sta, throughput)


class StaThroughputConfigEvent(events.EventBase):
    def __init__(self, sta, throughput):
        super().__init__()
        self.sta = sta
        self.throughput = throughput

    def serialize(self):
        return {"sta": self.sta, "throughput": self.throughput}

    @classmethod
    def parse(cls, buf):
        sta = buf.get("sta", None)
        throughput = buf.get("throughput", None)
        return cls(sta, throughput)


class StaPhyRateEvent(events.EventBase):
    def __init__(self, sta, phyRate):
        super().__init__()
        self.sta = sta
        self.phyRate = phyRate

    def serialize(self):
        return {"sta": self.sta, "phyRate": self.phyRate}

    @classmethod
    def parse(cls, buf):
        sta = buf.get("sta", None)
        phyRate = buf.get("phyRate", None)
        return cls(sta, phyRate)


class StaSlotShareEvent(events.EventBase):
    def __init__(self, sta, slotShare):
        super().__init__()
        self.sta = sta
        self.slotShare = slotShare

    def serialize(self):
        return {"sta": self.sta, "slotShare": self.slotShare}

    @classmethod
    def parse(cls, buf):
        sta = buf.get("sta", None)
        slotShare = buf.get("slotShare", None)
        return cls(sta, slotShare)
