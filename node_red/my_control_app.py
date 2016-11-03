import logging
import random
import wishful_upis as upis
from uniflex.core import modules
from uniflex.core import events
from uniflex.timer import TimerEventSender
from common import AveragedSpectrumScanSampleEvent

__author__ = "Piotr Gawlowicz"
__copyright__ = "Copyright (c) 2016, Technische Universität Berlin"
__version__ = "0.1.0"
__email__ = "{gawlowicz}@tkn.tu-berlin.de"


class PeriodicEvaluationTimeEvent(events.TimeEvent):
    def __init__(self):
        super().__init__()


@modules.build_module
class MyController(modules.ControllerModule):
    def __init__(self, mode):
        super(MyController, self).__init__()
        self.log = logging.getLogger('MyController')
        self.mode = mode
        self.running = False
        self.nodes = []

        self.timeInterval = 10
        self.timer = TimerEventSender(self, PeriodicEvaluationTimeEvent)
        self.timer.start(self.timeInterval)

        self.myFilterRunning = False
        self.packetLossEventsEnabled = False

    @modules.on_start()
    def my_start_function(self):
        print("start control app")
        self.running = True

    @modules.on_exit()
    def my_stop_function(self):
        print("stop control app")
        self.running = False

    @modules.on_event(events.NewNodeEvent)
    def add_node(self, event):
        node = event.node

        if self.mode == "GLOBAL" and node.local:
            return

        self.log.info("Added new node: {}, Local: {}"
                      .format(node.uuid, node.local))
        self.nodes.append(node)

        devs = node.get_devices()
        for dev in devs:
            print("Dev: ", dev.name)

        device = node.get_device(0)
        device.radio.set_tx_power(15, "wlan0")
        device.radio.set_channel(random.randint(1, 11))
        device.enable_event(upis.radio.PacketLossEvent)
        self.packetLossEventsEnabled = True
        device.start_service(
            upis.radio.SpectralScanService(rate=1000, f_range=[2200, 2500]))

    @modules.on_event(events.NodeExitEvent)
    @modules.on_event(events.NodeLostEvent)
    def remove_node(self, event):
        self.log.info("Node lost".format())
        node = event.node
        reason = event.reason
        if node in self.nodes:
            self.nodes.remove(node)
            self.log.info("Node: {}, Local: {} removed reason: {}"
                          .format(node.uuid, node.local, reason))

    @modules.on_event(upis.radio.PacketLossEvent)
    def serve_packet_loss_event(self, event):
        node = event.node
        device = event.device
        self.log.info("Packet loss in node {}, dev: {}"
                      .format(node.uuid, device.name))

    @modules.on_event(AveragedSpectrumScanSampleEvent)
    def serve_spectral_scan_sample(self, event):
        avgSample = event.avg
        self.log.info("Averaged Spectral Scan Sample: {}"
                      .format(avgSample))

    @modules.on_event(PeriodicEvaluationTimeEvent)
    def periodic_evaluation(self, event):
        # go over collected samples, etc....
        # make some decisions, etc...
        print("Periodic Evaluation")
        print("My nodes: ", [node.uuid for node in self.nodes])
        self.timer.start(self.timeInterval)

        if len(self.nodes) == 0:
            return
