import logging
import datetime

from uniflex.core import modules
from uniflex.core import events
from uniflex.core.timer import TimerEventSender

__author__ = "Piotr Gawlowicz, Mikołaj Chwalisz"
__copyright__ = "Copyright (c) 2016-2018, Technische Universität Berlin"
__version__ = "0.1.0"
__email__ = "{gawlowicz}@tkn.tu-berlin.de"


class PeriodicBlockingEvaluationTimeEvent(events.TimeEvent):
    def __init__(self):
        super().__init__()


class PeriodicNonBlockingEvaluationTimeEvent(events.TimeEvent):
    def __init__(self):
        super().__init__()


class SimpleBenchmark(modules.ControlApplication):
    def __init__(self, title=''):
        super(SimpleBenchmark, self).__init__()
        self.log = logging.getLogger('SimpleBenchmark')
        self.title = title
        self.running = False

        self.timeInterval = 4
        self.blocking_timer = TimerEventSender(
            self, PeriodicBlockingEvaluationTimeEvent)
        self.non_blocking_timer = TimerEventSender(
            self, PeriodicNonBlockingEvaluationTimeEvent)

        self.non_blocking_timer.start(self.timeInterval)

        self.packetLossEventsEnabled = False

    @modules.on_start()
    def my_start_function(self):
        self.log.info("start control app")
        print('"Experiment";"Call type";"Nr calls";"Total duration";"Average per call"')
        self.csv_template = '"{title}";{call};{nr};{total};{per_call}'
        self.running = True

    @modules.on_exit()
    def my_stop_function(self):
        self.log.info("stop control app")
        self.running = False

    @modules.on_event(events.NewNodeEvent)
    def add_node(self, event):
        node = event.node

        self.log.info("Added new node: {}, Local: {}"
                      .format(node.uuid, node.local))
        self._add_node(node)

    @modules.on_event(events.NodeExitEvent)
    @modules.on_event(events.NodeLostEvent)
    def remove_node(self, event):
        self.log.info("Node lost".format())
        node = event.node
        reason = event.reason
        if self._remove_node(node):
            self.log.info("Node: {}, Local: {} removed reason: {}"
                          .format(node.uuid, node.local, reason))

    def get_power_cb(self, data):

        self.current_num += 1

        if self.current_num >= self.repeatNum:
            end = datetime.datetime.now()
            duration = end - self.start
            perCall = duration / self.repeatNum
            self.log.info(
                "{} RPC calls were executed in: {}s".format(
                    self.repeatNum, duration))
            self.log.info(
                "--- mean duration of single call: {}s".format(perCall))
            print(self.csv_template.format(
                title=self.title,
                call='non blocking',
                nr=self.repeatNum,
                total=duration,
                per_call=perCall))

            self.blocking_timer.start(self.timeInterval)

    @modules.on_event(PeriodicNonBlockingEvaluationTimeEvent)
    def periodic_non_blocking_evaluation(self, event):
        self.log.info("Periodic Non Blocking Evaluation")
        self.log.info(
            "My nodes: %s",
            ', '.join([node.hostname for node in self.get_nodes()]))

        if len(self.get_nodes()) == 0:
            self.non_blocking_timer.start(self.timeInterval)
            return

        node = self.get_node(0)
        device = node.get_device(0)

        self.current_num = 0
        self.repeatNum = 10000
        self.start = datetime.datetime.now()
        self.log.info(
            "Start performace test, execute {} non blocking RPC calls".format(
                self.repeatNum))
        for i in range(self.repeatNum):
            # device.get_channel("wlan0")

            device.callback(self.get_power_cb).get_tx_power("wlan0")

    @modules.on_event(PeriodicBlockingEvaluationTimeEvent)
    def periodic_blocking_evaluation(self, event):
        self.log.info("Periodic Blocking Evaluation")
        self.log.info(
            "My nodes: %s",
            ', '.join([node.hostname for node in self.get_nodes()]))

        if len(self.get_nodes()) == 0:
            self.blocking_timer.start(self.timeInterval)
            return

        node = self.get_node(0)
        device = node.get_device(0)

        # test blocking call in loop
        self.current_num = 0
        self.repeatNum = 10000
        self.start = datetime.datetime.now()
        self.log.info(
            "Start performace test, execute {} blocking RPC calls".format(
                self.repeatNum))

        for i in range(self.repeatNum):
            device.get_channel("wlan0")

        end = datetime.datetime.now()
        duration = end - self.start
        perCall = duration / self.repeatNum
        self.log.info(
            "{} blocking RPC calls were executed in: {}s".format(
                self.repeatNum, duration))
        self.log.info(
            "--- mean duration of single blocking call: {}s".format(
                perCall))
        print(self.csv_template.format(
            title=self.title,
            call='blocking',
            nr=self.repeatNum,
            total=duration,
            per_call=perCall))

        self.non_blocking_timer.start(self.timeInterval)
