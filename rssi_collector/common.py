from uniflex.core import events

__author__ = "Piotr Gawlowicz"
__copyright__ = "Copyright (c) 2016, Technische Universität Berlin"
__version__ = "0.1.0"
__email__ = "{gawlowicz}@tkn.tu-berlin.de"


class AveragedRssiSampleEvent(events.EventBase):
    def __init__(self, receiverUuid, receiverDevId, ta, rssi):
        super().__init__()
        self.receiverUuid = receiverUuid
        self.receiverDevId = receiverDevId
        self.ta = ta
        self.rssi = rssi

    def serialize(self):
        return {"receiverUuid": self.receiverUuid,
                "receiverDevId": self.receiverDevId,
                "ta": self.ta, "rssi": self.rssi}

    @classmethod
    def parse(cls, buf):
        receiverUuid = buf.get("receiverUuid", None)
        receiverDevId = buf.get("receiverDevId", None)
        rssi = buf.get("rssi", None)
        ta = buf.get("ta", None)
        return cls(receiverUuid, receiverDevId, ta, rssi)


class StartMyFilterEvent(events.EventBase):
    def __init__(self):
        super().__init__()


class StopMyFilterEvent(events.EventBase):
    def __init__(self):
        super().__init__()
