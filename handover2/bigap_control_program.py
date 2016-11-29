import logging
import datetime
import wishful_upis as upis
from sbi.wifi.events import WiFiHandoverRequestEvent, WiFiGetServingAPRequestEvent, WiFiGetServingAPReplyEvent
from uniflex.core import modules
from uniflex.core import events
from uniflex.timer import TimerEventSender
from common import CQIReportingEvent
from common import DHCPNewEvent
from common import DHCPDelEvent

import time
import subprocess
import zmq

__author__ = "Anatolij Zubow"
__copyright__ = "Copyright (c) 2016, Technische Universität Berlin"
__version__ = "0.1.0"
__email__ = "{zubow}@tkn.tu-berlin.de"


class PeriodicSTADiscoveryTimeEvent(events.TimeEvent):
    def __init__(self):
        super().__init__()


'''
Set of control programs to be executed on central node, e.g. server:
(1) BigAP controller - makes handover decisions based on the CQI
                       reports of the APs. Gets information about
                       new clients from DHCP.
(2) DHCPDaemon - wraps DHCP server and informs BigAP controller
                 about new and removed leases.
'''

class BigAPController(modules.ControlApplication):
    """
    BigAP controller - makes handover decisions based on the CQI reports of the APs.
    Gets information about new clients from DHCP.
    """
    def __init__(self, mode, ap_iface):
        super(BigAPController, self).__init__()
        self.log = logging.getLogger('BigAPController')
        self.mode = mode
        self.ap_iface = ap_iface
        self.sta_discovery_interval = 5

        self.running = False
        self.activeSTAs = {}  # MAC_ADDR -> IP_ADDR
        self.nodes = {}  # APs UUID -> node
        self.servingAPs = {}  # STA_MAC_ADDR -> AP node


    @modules.on_start()
    def my_start_function(self):
        print("BiGAP control app started")

        # channel hopping every 100ms
        self.staDiscoveryTimer = TimerEventSender(self, PeriodicSTADiscoveryTimeEvent)
        self.staDiscoveryTimer.start(self.sta_discovery_interval)

        self.running = True


    @modules.on_exit()
    def my_stop_function(self):
        print("BiGAP control app stopped")
        self.running = False


    @modules.on_event(events.NewNodeEvent)
    def add_node(self, event):
        node = event.node

        if self.mode == "GLOBAL" and node.local:
            return

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


    @modules.on_event(PeriodicSTADiscoveryTimeEvent)
    def periodic_sta_discovery(self, event):
        if self.node is None:
            return

        self.log.debug("Periodic STA discovery")
        self.log.debug("My node: %s" % self.node.uuid)
        self.staDiscoveryTimer.start(self.sta_discovery_interval)

        try:
            active_sta_mac_addrs = list(self.nodes.keys())

            for sta_mac_addr_tmp in active_sta_mac_addrs:
                self.send_servingAP_req(sta_mac_addr_tmp, self.ap_iface)
        except Exception as e:
            self.log.error("{} !!!Exception!!!: {}".format(
                datetime.datetime.now(), e))


    @modules.on_event(CQIReportingEvent)
    def serve_cqi_report_event(self, event):
        '''
            From APs
        '''
        curr_sigpower = event.curr_sigpower
        candidate_sigpower = event.candidate_sigpower
        self.log.info("CQIReportingEvent curr: {}"
                      .format(curr_sigpower))
        self.log.info("CQIReportingEvent curr: {}"
                      .format(candidate_sigpower))

        # data structure: STA_MAC_ADDR -> dBm
        mac_addrs = list(self.curr_sigpower.keys())

        for sta_mac_addr in mac_addrs:
            print('tbd')
            pass


    @modules.on_event(DHCPNewEvent)
    def serve_dhcp_new_event(self, event):
        '''
            From DHCP
        '''
        self.log.info("DHCPNewEvent NEW: {}"
                      .format(event.mac_addr))
        self.log.info("DHCPNewEvent NEW: {}"
                      .format(event.ip_addr))

        if event.mac_addr not in self.activeSTAs:
            # new STA to be served
            self.activeSTAs[event.mac_addr] = event.ip_addr
        else:
            # already known
            pass


    @modules.on_event(DHCPDelEvent)
    def serve_dhcp_del_event(self, event):
        '''
            From DHCP
        '''
        self.log.info("DHCPNewEvent DEL: {}"
                      .format(event.mac_addr))
        self.log.info("DHCPNewEvent DEL: {}"
                      .format(event.ip_addr))

        if event.mac_addr in self.activeSTAs:
            # new STA to be served
            del self.activeSTAs[event.mac_addr]
        else:
            # unknown STA
            pass


    def send_servingAP_req(self, sta_mac_addr, iface):
        '''
            Functions send out a message to wireless topology
            app to discover the AP serving a particular client.
        '''
        try:
            self.log.debug('send_servingAP_req')

            ho_event = WiFiGetServingAPRequestEvent(sta_mac_addr, iface)
            self.send_event(ho_event)
        except Exception as e:
            self.log.fatal("... An error occurred : %s" % e)
            raise e


    @modules.on_event(WiFiGetServingAPReplyEvent)
    def rx_servingAP_reply(self, event):
        '''
            From wireless topology app.
        '''
        self.log.info("rx_servingAP_reply: {}"
                      .format(event))

        sta_mac_addr = event.sta_mac_addr
        wifi_intf = event.wifi_intf
        ap_uuids = event.ap_uuids

        # STA_MAC_ADDR -> AP node
        if ap_uuids in self.nodes:
            self.servingAPs[sta_mac_addr] = self.nodes[ap_uuids]
        else:
            self.log.error('Unknown ap_uuids %s' % ap_uuids)


    def trigger_handover(self, sta_mac_addr, serving_AP, target_AP, gateway, **kwargs):
        '''
            Functions triggers handover by sending
            a WiFiTriggerHandoverRequestEvent to the corresponding app.
        '''
        try:
            self.log.debug('performHO: send event to HO app')
            ho_event = WiFiHandoverRequestEvent(sta_mac_addr, serving_AP, target_AP, gateway, **kwargs)
            self.send_event(ho_event)
        except Exception as e:
            self.log.fatal("... An error occurred : %s" % e)
            raise e


    @modules.on_event(upis.net_func.TriggerHandoverReplyEvent)
    def handle_handover_reply(self, event):
        '''
            From Handover module
        '''
        if not event.success:
            self.log.error('Handover failed!')
        else:
            self.log.info('Handover done')


class DHCPDaemon(modules.ControlApplication):
    """
        DHCP daemon which sends events used for STA client discovery
    """
    def __init__(self, mode, zmq_port):
        super(DHCPDaemon, self).__init__()
        self.log = logging.getLogger('DHCPDaemon')
        self.mode = mode
        self.zmq_port = zmq_port
        # self.activeSTAs = {}
        self.running = False

    @modules.on_start()
    def my_start_function(self):
        print("BiGAP control app started")

        # start scanner
        self.exec_file = 'staticDHCPd'
        self.processArgs = self.exec_file + ' '
        self.process = subprocess.Popen(self.processArgs.split(), shell=False)
        # wait to settle down
        time.sleep(1)

        # ZMQ stuff
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.connect("tcp://localhost:%s" % self.zmq_port)

        self.running = True

        # ZMQ message loop
        while (True):
            dhcpJson = self.socket.recv_json()

            print(dhcpJson)

            for key in dhcpJson:
                print("key: %s , value: %s" % (key, dhcpJson[key]))
                if key == 'new':
                    print('NEW DHCP LEASE START THREAD')
                    dhcpNew = dhcpJson[key].split('/')
                    mac = dhcpNew[0]
                    ip = dhcpNew[1]

                    # if mac not in self.activeSTAs:
                    #    pass

                    event = DHCPNewEvent(mac, ip)
                    self.send_event(event)

                elif key == 'delete':
                    print('DHCP LEASE EXPIRED STOP THREAD')
                    dhcpDelete = dhcpJson[key].split('/')
                    mac = dhcpDelete[0]
                    ip = dhcpDelete[1]

                    event = DHCPDelEvent(mac, ip)
                    self.send_event(event)

            else:
                print('DHCP Server sent unknown key in dictionary')

    @modules.on_exit()
    def my_stop_function(self):
        print("BiGAP control app stopped")
        # stop scanner
        self.process.kill()
        self.running = False
