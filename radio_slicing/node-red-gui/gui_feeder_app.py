import logging
import random

from uniflex.core import modules
from uniflex.core import events
from uniflex.core.timer import TimerEventSender
from common import StaStateEvent, StaThroughputEvent, StaThroughputConfigEvent, StaPhyRateEvent, StaSlotShareEvent

__author__ = "Piotr Gawlowicz"
__copyright__ = "Copyright (c) 2016, Technische Universität Berlin"
__version__ = "0.1.0"
__email__ = "{gawlowicz}@tkn.tu-berlin.de"


class SampleSendTimeEvent(events.TimeEvent):
    def __init__(self):
        super().__init__()


class StateChangeTimeEvent(events.TimeEvent):
    def __init__(self):
        super().__init__()


class GuiFeeder(modules.ControlApplication):
    def __init__(self):
        super(GuiFeeder, self).__init__()
        self.log = logging.getLogger('GuiFeeder')

        self.sampleTimeInterval = 1
        self.sampleTimer = TimerEventSender(self, SampleSendTimeEvent)
        self.sampleTimer.start(self.sampleTimeInterval)

        self.stateChangeTimeInterval = 5
        self.stateChangeTimer = TimerEventSender(self, StateChangeTimeEvent)
        self.stateChangeTimer.start(self.stateChangeTimeInterval)

        self.sta_list = {"TV": False,
                         "MyLaptop": False,
                         "MySmartphone": False,
                         "Guest1": False,
                         "Guest2": False}

    @modules.on_event(StaThroughputConfigEvent)
    def serve_throughput_config_event(self, event):
        sta = event.sta
        throughput = event.throughput
        self.log.info("Set Thrughput for STA: {}: to: {}"
                      .format(sta, throughput))

    @modules.on_event(StateChangeTimeEvent)
    def change_sta_state(self, event):
        # reschedule function
        self.stateChangeTimer.start(self.stateChangeTimeInterval)

        mySTAs = list(self.sta_list.keys())
        sta = random.choice(mySTAs)
        # get current state
        currentState = self.sta_list[sta]
        # toogle state
        newState = not currentState
        event = StaStateEvent(sta, newState)
        self.send_event(event)
        # update my list
        self.sta_list[sta] = newState
        self.log.info("Change state of STA: {}: new state:{}"
                      .format(sta, newState))

    @modules.on_event(SampleSendTimeEvent)
    def send_random_samples(self, event):
        # reschedule function
        self.sampleTimer.start(self.sampleTimeInterval)
        # send random data to GuiFeeder

        for sta, state in self.sta_list.items():
            if state:
                self.log.info("Send new random samples for device: {}"
                              .format(sta))
                throughput = random.uniform(1, 30)
                event = StaThroughputEvent(sta, throughput)
                self.send_event(event)

                phyRate = random.uniform(5, 54)
                event = StaPhyRateEvent(sta, phyRate)
                self.send_event(event)

                slotShare = random.uniform(0, 100)
                event = StaSlotShareEvent(sta, slotShare)
                self.send_event(event)
