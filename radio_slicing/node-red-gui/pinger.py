import logging
import subprocess as sp

from uniflex.core import modules
from uniflex.core import events
from uniflex.core.timer import TimerEventSender
from common import HostStateEvent

__author__ = "Piotr Gawlowicz"
__copyright__ = "Copyright (c) 2016, Technische Universität Berlin"
__version__ = "0.1.0"
__email__ = "{gawlowicz}@tkn.tu-berlin.de"


class PingTimeEvent(events.TimeEvent):
    def __init__(self):
        super().__init__()


class Pinger(modules.ControlApplication):
    def __init__(self, hostList=[], interval=1):
        super(Pinger, self).__init__()
        self.log = logging.getLogger('Pinger')

        self.pingInterval = interval
        self.pingTimer = TimerEventSender(self, PingTimeEvent)
        self.pingTimer.start(self.pingInterval)

        self.hostList = hostList

    def _ping(self, address):
        status, result = sp.getstatusoutput("ping -c1 -w1 " + str(address))
        if status == 0:
            return True
        else:
            return False

    @modules.on_event(PingTimeEvent)
    def send_random_samples(self, event):
        # reschedule function
        self.pingTimer.start(self.pingInterval)

        # ping devices
        for host in self.hostList:
            isUp = self._ping(host)
            if isUp:
                self.log.info("Host {} is UP".format(host))
            else:
                self.log.info("Host {} is DOWN".format(host))

            event = HostStateEvent(host, isUp)
            self.send_event(event)
