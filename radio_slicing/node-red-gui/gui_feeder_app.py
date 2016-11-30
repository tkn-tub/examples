import logging
import random

from uniflex.core import modules
from uniflex.core import events
from uniflex.core.timer import TimerEventSender
from common import StaInfo, IperfClientProcess, HostStateEvent
from common import StaStateEvent, StaThroughputEvent
from common import StaThroughputConfigEvent, StaPhyRateEvent, StaSlotShareEvent


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
    def __init__(self, staList=[]):
        super(GuiFeeder, self).__init__()
        self.log = logging.getLogger('GuiFeeder')

        self.sta_list = []
        for sta in staList:
            self.log.debug("New STA: {},{},{}".format(sta[0], sta[1], sta[2]))
            s = StaInfo(sta[0], sta[1], sta[2])
            self.sta_list.append(s)

    @modules.on_event(StaThroughputConfigEvent)
    def serve_throughput_config_event(self, event):
        sta = event.sta
        throughput = event.throughput
        self.log.info("Set Thrughput for STA: {}: to: {}"
                      .format(sta, throughput))

    @modules.on_event(HostStateEvent)
    def change_sta_state(self, event):
        self.log.info("Host: {} state info".format(event.ip,))

        for s in self.sta_list:
            if s.ip == event.ip:
                if s.state is None:
                    s.state = event.state
                    return
                elif ((s.state and not event.state) or
                      (not s.state and event.state)):
                    self.log.info("STA {}:{} changed state to {}"
                                  .format(s.name, s.ip, event.state))
                    s.state = event.state

                    # sent event to GUI
                    ev = StaStateEvent(s.name, s.state)
                    self.send_event(ev)

                    # if new state if UP start iperf
                    if s.state:
                        if s.iperfProcess is None:
                            s.iperfProcess = IperfClientProcess(self, s.name, s.ip)
                        s.iperfProcess.start()
                    else:
                        if s.iperfProcess:
                            s.iperfProcess.stop()

    @modules.on_event(SampleSendTimeEvent)
    def send_random_samples(self, event):
        for sta in self.sta_list:
            if sta.state:
                self.log.info("Send new random samples for device: {}"
                              .format(sta.name))

                phyRate = random.uniform(5, 54)
                event = StaPhyRateEvent(sta.name, phyRate)
                # self.send_event(event)

                slotShare = random.uniform(0, 100)
                event = StaSlotShareEvent(sta.name, slotShare)
                # self.send_event(event)
