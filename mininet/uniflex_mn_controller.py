import logging
import datetime
import pprint
from uniflex.core import modules
from uniflex.core import events
from uniflex.timer import TimerEventSender

__author__ = "Anatolij Zubow"
__copyright__ = "Copyright (c) 2016, Technische Universität Berlin"
__version__ = "0.1.0"
__email__ = "{zubow}@tkn.tu-berlin.de"


class PeriodicTimeEvent(events.TimeEvent):
    def __init__(self):
        super().__init__()


'''
Simple control program for testing mininet functionality.
'''


class MininetWiFiController(modules.ControlApplication):
    def __init__(self):
        super(MininetWiFiController, self).__init__()
        self.log = logging.getLogger('MininetWiFiController')
        self.interval = 5
        self.nodes = {}  # APs UUID -> node
        self.running = False

    @modules.on_start()
    def my_start_function(self):
        self.log.info("Topology control app started")

        # channel hopping every 100ms
        self.staTimer = TimerEventSender(self, PeriodicTimeEvent)
        self.staTimer.start(self.interval)

        self.running = True

    @modules.on_exit()
    def my_stop_function(self):
        self.running = False

    @modules.on_event(events.NewNodeEvent)
    def add_node(self, event):
        node = event.node

        self.log.info("Added new node: {}"
                      .format(node.uuid))
        self.nodes[node.uuid] = node

    @modules.on_event(events.NodeExitEvent)
    @modules.on_event(events.NodeLostEvent)
    def remove_node(self, event):
        self.log.info("Node lost".format())
        node = event.node
        reason = event.reason
        if node.uuid in self.nodes:
            del self.nodes[node.uuid]
            self.log.info("Node: {}, removed reason: {}"
                          .format(node.uuid, reason))

    @modules.on_event(PeriodicTimeEvent)
    def periodic_tx_stats(self, event):

        self.log.info("Periodic getting tx stats")
        self.staTimer.start(self.interval)

        try:

            ap_uuids = list(self.nodes.keys())

            for ap_uuid in ap_uuids:
                ap_node = self.nodes[ap_uuid]

                if ap_node.name == "AP1":
                    node_iface = 'ap1-wlan0'
                elif ap_node.name == "AP2":
                    node_iface = 'ap2-wlan0'

                self.log.info("Querying node ... %s for iface: %s" % (ap_node.name, node_iface))
                ap_sta = ap_node.net.get_tx_bytes_of_connected_devices(node_iface)

                self.log.info('STAs of AP:\n{}'.format(pprint.pformat(ap_sta)))

        except Exception as e:
            self.log.error("{} !!!Exception!!!: {}".format(
                datetime.datetime.now(), e))
