import logging
import datetime
import wishful_upis as upis
from wishful_agent.core import wishful_module
from wishful_agent.core import events
from wishful_agent.timer import TimerEventSender

__author__ = "Anatolij Zubow"
__copyright__ = "Copyright (c) 2016, Technische Universität Berlin"
__version__ = "0.1.0"
__email__ = "{zubow}@tkn.tu-berlin.de"


class PeriodicSTADiscoveryTimeEvent(events.TimeEvent):
    def __init__(self):
        super().__init__()


'''
Simple control program for topology discovery.
'''


@wishful_module.build_module
class WiFiTopologyController(wishful_module.ControllerModule):
    def __init__(self, mode, ap_iface):
        super(WiFiTopologyController, self).__init__()
        self.log = logging.getLogger('WiFiTopologyController')
        self.mode = mode
        self.ap_iface = ap_iface
        self.sta_discovery_interval = 5

        self.running = False
        self.nodes = {}  # APs UUID -> node
        self.active_sta_mac_addrs = ['00:11:22:33:44:55']

    @wishful_module.on_start()
    def my_start_function(self):
        self.log.info("Topology control app started")

        # channel hopping every 100ms
        self.staDiscoveryTimer = TimerEventSender(self, PeriodicSTADiscoveryTimeEvent)
        self.staDiscoveryTimer.start(self.sta_discovery_interval)

        self.running = True

    @wishful_module.on_exit()
    def my_stop_function(self):
        self.log.info("Topology control app stopped")
        self.running = False

    @wishful_module.on_event(events.NewNodeEvent)
    def add_node(self, event):
        node = event.node

        self.log.info("Added new node: {}"
                      .format(node.uuid))
        self.nodes[node.uuid] = node

    @wishful_module.on_event(events.NodeExitEvent)
    @wishful_module.on_event(events.NodeLostEvent)
    def remove_node(self, event):
        self.log.info("Node lost".format())
        node = event.node
        reason = event.reason
        if node in self.nodes:
            del self.nodes[node.uuid]
            self.log.info("Node: {}, removed reason: {}"
                          .format(node.uuid, reason))

    @wishful_module.on_event(PeriodicSTADiscoveryTimeEvent)
    def periodic_sta_discovery(self, event):

        self.log.info("Periodic STA discovery")
        self.staDiscoveryTimer.start(self.sta_discovery_interval)

        try:
            for sta_mac_addr_tmp in self.active_sta_mac_addrs:
                ho_event = upis.wifi.WiFiGetServingAPRequestEvent(sta_mac_addr_tmp, self.ap_iface)
                self.log.info("... send event for %s " % sta_mac_addr_tmp)
                self.send_event(ho_event)

        except Exception as e:
            self.log.error("{} !!!Exception!!!: {}".format(
                datetime.datetime.now(), e))

    @wishful_module.on_event(upis.wifi.WiFiGetServingAPReplyEvent)
    def rx_serving_reply(self, event):

        if event.ap_uuid in self.nodes:
            node = self.nodes[event.ap_uuid]

        self.log.info("RX WiFiGetServingAPReplyEvent: %s, %s, %s, %s" % (event.sta_mac_addr, event.wifi_intf, event.ap_uuid, node.hostname))
