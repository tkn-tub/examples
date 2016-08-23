import logging
import wishful_upis as upis
import wishful_framework as wishful_module
from .common import AveragedSpectrumScanSampleEvent


__author__ = "Piotr Gawlowicz"
__copyright__ = "Copyright (c) 2016, Technische Universität Berlin"
__version__ = "0.1.0"
__email__ = "{gawlowicz}@tkn.tu-berlin.de"


@wishful_module.build_module
class MyAvgFilter(wishful_module.ControllerModule):
    def __init__(self, window):
        super(MyAvgFilter, self).__init__()
        self.log = logging.getLogger('MyFilter')
        self.window = window
        self.running = False
        self.nodes = []
        self.samples = []

    @wishful_module.on_start()
    def my_start_function(self):
        print("start control app")
        self.running = True

    @wishful_module.on_exit()
    def my_stop_function(self):
        print("stop control app")
        self.running = False

    @wishful_module.on_event(upis.radio.SpectralScanSampleEvent)
    def serve_spectral_scan_sample(self, event):
        sample = event.sample
        node = event.node
        device = event.device
        self.log.debug("New SpectralScan Sample:{} from node {}, device: {}"
                       .format(sample, node, device))

        self.samples.append(sample)

        if len(self.samples) == self.window:
            s = sum(self.samples)
            self.samples.pop(0)
            avg = s / self.window
            self.log.debug("Calculated average: {}".format(avg))
            event = AveragedSpectrumScanSampleEvent(avg)
            self.send_event(event)