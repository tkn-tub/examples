import logging
import wishful_upis as upis
from uniflex.core import modules
from uniflex.core import events
from uniflex.timer import TimerEventSender
from uniflex.node import Node, Device
from common import AveragedRssiSampleEvent


__author__ = "Piotr Gawlowicz"
__copyright__ = "Copyright (c) 2016, Technische Universität Berlin"
__version__ = "0.1.0"
__email__ = "{gawlowicz}@tkn.tu-berlin.de"


class PeriodicEvaluationTimeEvent(events.TimeEvent):
    def __init__(self):
        super().__init__()


class MyController(modules.ControlApplication):
    def __init__(self):
        super(MyController, self).__init__()
        self.log = logging.getLogger('MyController')
        self.running = False
        self.nodes = []

        self.timeInterval = 10
        self.timer = TimerEventSender(self, PeriodicEvaluationTimeEvent)
        self.timer.start(self.timeInterval)

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
        self.log.info("Added new node: {}, Local: {}"
                      .format(node.uuid, node.local))
        self._add_node(node)

        for device in node.get_devices():
            print("Dev: ", device.name)
            device.radio.add_interface('mon0', 'monitor')
            device.start_service(
                upis.radio.RssiService)

    @modules.on_event(events.NodeExitEvent)
    @modules.on_event(events.NodeLostEvent)
    def remove_node(self, event):
        self.log.info("Node lost".format())
        node = event.node
        reason = event.reason
        if self._remove_node(node):
            self.log.info("Node: {}, Local: {} removed reason: {}"
                          .format(node.uuid, node.local, reason))

    @modules.on_event(AveragedRssiSampleEvent)
    def serve_spectral_scan_sample(self, event):
        receiver = event.receiverUuid
        if self._get_node_by_uuid(event.receiverUuid):
            receiver = self._get_node_by_uuid(event.receiverUuid)

        dev = event.receiverDevId
        if receiver:
            dev = receiver.get_device(event.receiverDevId)

        ta = event.ta
        avgSample = event.rssi

        if isinstance(receiver, Node) and isinstance(dev, Device):
            self.log.info("Avg RSSI: receiver: {}:{}, TA: {}, value: {}"
                          .format(receiver.hostname, dev.name, ta, avgSample))
        else:
            self.log.info("Avg RSSI: receiver: {}:{}, TA: {}, value: {}"
                          .format(receiver, dev, ta, avgSample))

    @modules.on_event(PeriodicEvaluationTimeEvent)
    def periodic_evaluation(self, event):
        # go over collected samples, etc....
        # make some decisions, etc...
        print("Periodic Evaluation")
        print("My nodes: ", [node.hostname for node in self.get_nodes()])
        self.timer.start(self.timeInterval)

        if len(self.get_nodes()) == 0:
            return
