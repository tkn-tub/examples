import time
import logging
import datetime
import subprocess

from uniflex.core import modules
from uniflex.core import events
from uniflex.core.events import TimeEvent
from uniflex.core.timer import TimerEventSender

from .common import CQIReportingEvent

__author__ = "Anatolij Zubow"
__copyright__ = "Copyright (c) 2016, Technische Universität Berlin"
__version__ = "0.1.0"
__email__ = "{zubow}@tkn.tu-berlin.de"

'''
Set of control programs to be executed at each AP locally:
(1) Scanner for reporting handover opportunities
'''


class PeriodicChannelSwitchTimeEvent(TimeEvent):
    def __init__(self):
        super().__init__()


class PeriodicCQIReportingTimeEvent(TimeEvent):
    def __init__(self):
        super().__init__()


"""
    Local control program running on each AP node to collect
    information about current link signal power and towards
    candidate APs using passive scanning.

    @return it announces the results using periodic events;
    i.e. CQIReportingEvent
"""


class Scanner(modules.ControlApplication):
    def __init__(self, mode, ap_iface, scan_iface, channels,
                 hopping_interval, reporting_interval):
        super(Scanner, self).__init__()
        self.log = logging.getLogger('Scanner')
        self.mode = mode
        self.ap_iface = ap_iface
        self.scan_iface = scan_iface
        self.channels = channels
        self.hopping_interval = hopping_interval
        self.reporting_interval = reporting_interval
        self.sta_map_file = '/tmp/sta_map.dat'
        self.next_ch_id = 0


    @modules.on_start()
    def my_start_function(self):
        self.log.debug("start scanner app")

        # start scanner
        self.exec_file = 'scanner/scanner'
        self.processArgs = self.exec_file + ' ' + self.scan_iface
        self.process = subprocess.Popen(self.processArgs.split(), shell=False)
        # wait to settle down
        time.sleep(1)

        # channel hopping every 100ms
        self.chHoppingTimeInterval = self.hopping_interval
        self.chTimer = TimerEventSender(self, PeriodicChannelSwitchTimeEvent)
        self.chTimer.start(self.chHoppingTimeInterval)

        # reporting signal quality as event
        self.reportingTimeInterval = self.reporting_interval
        self.reportingTimer = TimerEventSender(self, PeriodicCQIReportingTimeEvent)
        self.reportingTimer.start(self.reportingTimeInterval)


    @modules.on_exit()
    def my_stop_function(self):
        self.log.debug("stop scanner app")
        # stop scanner
        self.process.kill()


    @modules.on_event(PeriodicChannelSwitchTimeEvent)
    def periodic_channel_switch(self, event):

        self.log.debug("Periodic channel hopping")
        self.log.debug("My node: %s" % self.localNode.hostname)
        self.chTimer.start(self.chHoppingTimeInterval)

        device = self.localNode.get_device(0)

        try:
            # switch to next channel
            device.radio.set_channel(self.channels[self.next_ch_id ])

            self.next_ch_id = (self.next_ch_id + 1) % len(self.channels)
        except Exception as e:
            self.log.error("{} !!!Exception!!!: {}".format(
                datetime.datetime.now(), e))


    @modules.on_event(PeriodicCQIReportingTimeEvent)
    def periodic_reporting(self, event):

        self.log.debug("Periodic reporting")
        self.log.debug("My node: %s" % self.localNode.hostname)
        self.chTimer.start(self.reportingTimeInterval)

        device = self.localNode.get_device(0)

        try:
            # read and create event
            # CQI towards co-located client not being served
            candidate_sigpower = self.read_passive_scan_results()
            # CQI towards served client
            curr_sigpower = self.get_avg_sigpower()

            event = CQIReportingEvent(candidate_sigpower, curr_sigpower)

            self.send_event(event)
        except Exception as e:
            self.log.error("{} !!!Exception!!!: {}".format(
                datetime.datetime.now(), e))


    def read_passive_scan_results(self):
        """
        Get the signal quality towards co-located clients being served by neighboring APs. STA_MAC_ADDR -> dBm
        Returns in dBm
        """
        scan_results = open(self.sta_map_file, 'r')
        rv = {}
        for line in scan_results:
            row_arr = line.split(",")
            if (len(row_arr) >= 2):
                macAddr = row_arr[0]
                sigPow = row_arr[1]
            else:
                continue
            rv[macAddr] = sigPow
        return rv


    def get_avg_sigpower(self):
        """
        Get the signal quality of currently served/associated client stations. STA_MAC_ADDR -> dBm
        Returns in dBm
        """
        rv = {}
        station_info = self.localNode.net.iface(self.ap_iface).get_info_of_connected_devices()
        if station_info is not None:
            for node in station_info.keys():
                agent_control_ip = node.ip
                station_info_node = station_info[node]
                for mac in station_info_node.keys():
                    rv[mac] = station_info[node][mac]['signal avg'][0]
        return rv
