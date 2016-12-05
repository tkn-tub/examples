import logging
import datetime
import time

from uniflex.core import modules
from uniflex.core import events
from uniflex.core.timer import TimerEventSender

__author__ = "Anatolij Zubow"
__copyright__ = "Copyright (c) 2016, Technische Universität Berlin"
__version__ = "0.1.0"
__email__ = "{zubow}@tkn.tu-berlin.de"


class PeriodicEvaluationTimeEvent(events.TimeEvent):
    def __init__(self):
        super().__init__()


'''
	Global controller performs coordinated channel hopping and channel sounding
	in 802.11 network using Intel 5300 chipsets.
'''

class ChannelSounderWiFiController(modules.ControlApplication):
    def __init__(self, num_nodes):
        super(ChannelSounderWiFiController, self).__init__()
        self.log = logging.getLogger('ChannelSounderWiFiController')
        self.log.info("ChannelSounderWiFiController")
        self.nodes = {}  # APs UUID -> node
        self.num_nodes = num_nodes

    @modules.on_start()
    def my_start_function(self):
        self.log.info("start control app")

        self.next_channel = 1
        self.ifaceName = 'mon0'
        self.start = None
        self.hopping_interval = 1

        self.timeInterval = 0.5
        self.timer = TimerEventSender(self, PeriodicEvaluationTimeEvent)
        self.timer.start(self.timeInterval)

    def schedule_ch_switch(self, node=None):
        try:
            self.log.info('schedule_ch_switch')
            # schedule first channel switch in now + 3 seconds
            if self.start == None:
                self.start = time.time() + 3
            else:
                self.start = self.start + self.hopping_interval

            if node:
                device = node.get_device(0)
                device.exec_time(self.start).callback(self.channel_set_cb).set_channel(self.next_channel, self.ifaceName)
                #device.exec_time(self.start).callback(self.channel_set_cb).debug(self.next_channel)
            else:
                for node in self.nodes.values():
                    device = node.get_device(0)
                    device.exec_time(self.start).callback(self.channel_set_cb).set_channel(self.next_channel, self.ifaceName)
                    #device.exec_time(self.start).callback(self.channel_set_cb).debug(self.next_channel)

        except Exception as e:
            self.log.error("{} !!!Exception!!!: {}".format(
                datetime.datetime.now(), e))

    def channel_set_cb(self, data):
        node = data.node

        # schedule new ch switch
        self.schedule_ch_switch(node)

    @modules.on_exit()
    def my_stop_function(self):
        print("stop control app")

    @modules.on_event(events.NewNodeEvent)
    def add_node(self, event):
        node = event.node

        self.log.info("Added new node: {}, Local: {}"
                      .format(node.uuid, node.local))
        self.nodes[node.uuid] = node

        devs = node.get_devices()
        for dev in devs:
            self.log.info("Dev: ", dev.name)

    @modules.on_event(events.NodeExitEvent)
    @modules.on_event(events.NodeLostEvent)
    def remove_node(self, event):
        self.log.info("Node lost".format())
        node = event.node
        reason = event.reason
        if node in self.nodes:
            del self.nodes[node.uuid]
            self.log.info("Node: {}, Local: {} removed reason: {}"
                          .format(node.uuid, node.local, reason))


    @modules.on_event(PeriodicEvaluationTimeEvent)
    def periodic_evaluation(self, event):
        print("all node are available ...")
        if len(self.nodes) < self.num_nodes:
            # wait again
            self.timer.start(self.timeInterval)
        else:
            self.schedule_ch_switch()
