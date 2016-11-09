import logging
from collections import defaultdict

import wishful_upis as upis
from uniflex.core import modules
from common import AveragedRssiSampleEvent

__author__ = "Piotr Gawlowicz"
__copyright__ = "Copyright (c) 2016, Technische Universität Berlin"
__version__ = "0.1.0"
__email__ = "{gawlowicz}@tkn.tu-berlin.de"


class MyAvgFilter(modules.ControlApplication):
    def __init__(self, window):
        super(MyAvgFilter, self).__init__()
        self.log = logging.getLogger('MyAvgFilter')
        self.window = window
        self.samplesPerTransmitter = {}
        # rssi [receiverUUID][TA] = [samples]
        self.rssi = defaultdict(dict)

    def get_samples(self, receiverUuid, ta):
        try:
            return self.rssi[receiverUuid][ta]
        except:
            return None

    def add_sample(self, receiverUuid, ta, sample):
        if self.get_samples(receiverUuid, ta):
            self.rssi[receiverUuid][ta].append(sample)
        else:
            self.rssi[receiverUuid][ta] = [sample]

    def pop_samples(self, receiverUuid, ta, sampleNum=1):
        for i in range(sampleNum):
            self.rssi[receiverUuid][ta].pop(0)

    @modules.on_event(upis.radio.RssiSampleEvent)
    def serve_rssi_sample(self, event):

        rssi = event.rssi
        ta = event.ta
        node = event.node
        device = event.device
        self.log.debug(
            "RSSI Sample: node: {}, device: {}, transmitter: {}, value: {}"
            .format(node.hostname, device.name, ta, rssi))

        self.add_sample(node.uuid, ta, rssi)
        samples = self.get_samples(node.uuid, ta)

        if len(samples) == self.window:
            s = sum(samples)
            self.pop_samples(node.uuid, ta, 25)
            avg = s / self.window
            event = AveragedRssiSampleEvent(node.uuid, device._id, ta, avg)
            self.send_event(event)
            self.log.debug(
                "Averaged RSSI Sample: transmitter: {}, value: {}"
                .format(ta, avg))
