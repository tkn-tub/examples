import os
import logging
import datetime
import random
from uniflex.core import modules
from uniflex.core import events
from uniflex.core.timer import TimerEventSender
from uniflex_app_iperf import IperfServerRequestEvent, IperfClientRequestEvent

__author__ = "Anatolij Zubow"
__copyright__ = "Copyright (c) 2016, Technische Universität Berlin"
__version__ = "0.1.0"
__email__ = "{zubow}@tkn.tu-berlin.de"


class PeriodicEvaluationTimeEvent(events.TimeEvent):
    def __init__(self):
        super().__init__()


"""
    Simple control program testing iperf app.
"""
class MyIperfController(modules.ControlApplication):
    def __init__(self):
        super(MyIperfController, self).__init__()
        self.log = logging.getLogger('MyIperfController')


    @modules.on_start()
    def my_start_function(self):
        self.log.info("start control app")

        node = self.localNode

        self.log.debug("My local node: {}, Local: {}"
                      .format(node.hostname, node.local))

        for dev in node.get_devices():
            self.log.debug("Dev: %s" % dev.name)

        for m in node.get_modules():
            self.log.debug("Module: %s" % m.name)

        for apps in node.get_control_applications():
            self.log.debug("App: %s" % apps.name)

        self.device = node.get_device(0)

        # start iperf server
        iperfServerEvent = IperfServerRequestEvent()
        #iperfServerEvent.resultReportInterval = 1
        iperfServerEvent.stopAfterFirstReport = True

        self.log.info("Start iperf server ...")
        self.send_event(iperfServerEvent)

        self.timeInterval = 1
        self.timer = TimerEventSender(self, PeriodicEvaluationTimeEvent)
        self.timer.start(self.timeInterval)


    @modules.on_exit()
    def my_stop_function(self):
        print("stop control app")


    @modules.on_event(PeriodicEvaluationTimeEvent)
    def periodic_evaluation(self, event):
        self.log.info("Periodic channel hopping ...")

        # start iperf server
        iperfClientEvent = IperfClientRequestEvent()
        #iperfClientEvent.resultReportInterval = 1
        #iperfClientEvent.stopAfterFirstReport = True
        iperfClientEvent.transmissionTime = 5

        self.log.info("Start iperf client ...")
        self.send_event(iperfClientEvent)

        #self.timer.start(self.timeInterval)
