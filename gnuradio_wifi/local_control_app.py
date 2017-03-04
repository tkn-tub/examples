import os 
import logging 
import datetime 
import random 
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


"""
    Simple control program controlling a network node running GnuRadio WiFi (gr-802.11) to perform slow channel hopping.
"""
class MyGnuRadioWiFiController(modules.ControlApplication):
    def __init__(self):
        super(MyGnuRadioWiFiController, self).__init__()
        self.log = logging.getLogger('MyGnuRadioWiFiController')
        self.min_ch = 36
        self.max_ch = 52
        self.step_ch = 4
        self.ch = self.min_ch

    @modules.on_start()
    def my_start_function(self):
        self.log.info("start control app")

        node = self.localNode
        self.device = node.get_device(0)

        #self.timeInterval = 1
        #self.timer = TimerEventSender(self, PeriodicEvaluationTimeEvent)
        #self.timer.start(self.timeInterval)

    @modules.on_exit()
    def my_stop_function(self):
        print("stop control app")

        self.device.deactivate_radio_program()

    @modules.on_event(PeriodicEvaluationTimeEvent)
    def periodic_evaluation(self, event):
        print("Periodic channel hopping ...")

        self.ch = self.ch + self.step_ch
        if self.ch > self.max_ch:
            self.ch = self.min_ch

        # set new channel
        #self.device.set_channel(self.ch)
        #self.log.info("New freq is: %s" % self.ch)

        #self.timer.start(self.timeInterval)
