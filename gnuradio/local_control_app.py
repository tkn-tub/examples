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
    Simple control program controlling a network node running GnuRadio to perform slow channel hopping.
"""
class MyGnuRadioController(modules.ControlApplication):
    def __init__(self):
        super(MyGnuRadioController, self).__init__()
        self.log = logging.getLogger('MyGnuRadioController')
        self.running = False

        # GnuRadio program to be loaded
        fid = open(os.path.join(os.path.expanduser("."), "testgrc.grc"))
        self.grc_xml = fid.read()

        self.min_freq = 5200e6
        self.max_freq = 5300e6
        self.step_freq = 10e6
        self.freq = self.min_freq

    @modules.on_start()
    def my_start_function(self):
        self.log.info("start control app")
        self.running = True

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

        # activate GnuRadio RP
        self.grc_radio_program_name = 'test'
        inval = {}
        inval['ID'] = 11
        inval['grc_radio_program_code'] = self.grc_xml

        self.device.activate_radio_program(self.grc_radio_program_name, **inval)

        self.timeInterval = 1
        self.timer = TimerEventSender(self, PeriodicEvaluationTimeEvent)
        self.timer.start(self.timeInterval)

    @modules.on_exit()
    def my_stop_function(self):
        print("stop control app")
        tvals = {}
        tvals['do_pause'] = str(False)

        self.device.deactivate_radio_program(self.grc_radio_program_name, **tvals)

        self.running = False

    @modules.on_event(PeriodicEvaluationTimeEvent)
    def periodic_evaluation(self, event):
        print("Periodic channel hopping ...")

        self.freq = self.freq + self.step_freq
        if self.freq > self.max_freq:
            self.freq = self.min_freq

        inval = {}
        inval['freq'] = self.freq
        # set new freq
        self.device.set_parameters(inval)
        self.log.info("New freq is: %s" % self.freq)

        self.timer.start(self.timeInterval)
