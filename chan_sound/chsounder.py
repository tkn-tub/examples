import logging
import datetime
import datetime

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

        self.channel_lst = (1, 3, 6, 9, 11)
        self.next_channel_idx = 1
        self.ifaceName = 'mon0'
        self.start = None
        #self.hopping_interval = 3
        self.margot_uuid = None

        # CSI stuff
        self.results = []

        self.timeInterval = 1
        self.timer = TimerEventSender(self, PeriodicEvaluationTimeEvent)
        self.timer.start(self.timeInterval)

    def schedule_ch_switch(self):
        try:
            # schedule first channel switch in now + 3 seconds
            #if self.start == None:
            #    self.start = datetime.datetime.now() + datetime.timedelta(seconds=3)
            #else:
            #    self.start = self.start + datetime.timedelta(seconds=self.hopping_interval)

            # new channel
            self.next_channel_idx = (self.next_channel_idx + 1) % len(self.channel_lst)
            nxt_channel = self.channel_lst[self.next_channel_idx]

            self.log.info('schedule_ch_switch at %s to %d' % (str(self.start), nxt_channel))

            for node in self.nodes.values():
                device = node.get_device(0)
                #device.exec_time(self.start).callback(self.channel_set_cb).set_channel(nxt_channel, self.ifaceName)
                if node.uuid == self.margot_uuid:
                    device.callback(self.channel_set_cb).set_channel(nxt_channel, self.ifaceName)
                else:
                    device.callback(self.channel_set_cb).set_channel(nxt_channel, 'wlan0')

        except Exception as e:
            self.log.error("{} !!!Exception!!!: {}".format(
                datetime.datetime.now(), e))


    def channel_set_cb(self, data):
        """
        Callback function called when channel switching is done
        """
        node = data.node
        device = node.get_device(0)
        samples = 1
        csi = device.get_csi(samples, False)

        tuple = (self.channel_lst[self.next_channel_idx], node.uuid, csi)

        self.results.append(tuple)

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
            self.log.info("Dev: %s" % str(dev.name))
            ifaces = dev.get_interfaces()
            self.log.info('Ifaces %s' % ifaces)
            if 'mon0' in ifaces:
                self.margot_uuid = node.uuid

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
        self.timer.start(self.timeInterval)

        if len(self.nodes) < self.num_nodes:
            # wait again
            pass
        else:
            self.schedule_ch_switch()

