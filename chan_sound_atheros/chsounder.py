import logging
import datetime
import datetime
import numpy as np

from uniflex.core import modules
from uniflex.core import events

__author__ = "Anatolij Zubow"
__copyright__ = "Copyright (c) 2016, Technische Universität Berlin"
__version__ = "0.1.0"
__email__ = "{zubow}@tkn.tu-berlin.de"


'''
	Global controller performs channel sounding in 802.11 network using Atheros WiFi chipsets supporting
	CSI.
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

        self.ifaceName = 'wlan0'
        self.start = None
        #self.hopping_interval = 3

        # CSI stuff
        self.results = []
        self.samples = 1


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


        if len(self.nodes) == self.num_nodes:
            self.schedule_fetch_csi()


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


    def schedule_fetch_csi(self):
        try:
            self.log.info('First schedule_fetch_csi')

            for node in self.nodes.values():
                device = node.get_device(0)
                device.callback(self.channel_csi_cb).get_csi(self.samples, False)

        except Exception as e:
            self.log.error("{} !!!Exception!!!: {}".format(
                datetime.datetime.now(), e))


    def channel_csi_cb(self, data):
        """
        Callback function called when CSI results are available
        """
        node = data.node
        devName = None
        if data.device:
            devName = data.device.name
        csi = data.msg

        print("Default Callback: "
              "Node: {}, Dev: {}, Data: {}"
              .format(node.hostname, devName, csi.shape))

        csi_0 = csi[0].view(np.recarray)

        print(csi_0.header)
        #print(csi_0.csi_matrix)
        self.results.append(csi_0)

        # schedule callback for next CSI value
        data.device.callback(self.channel_csi_cb).get_csi(self.samples, False)

