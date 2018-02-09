import logging

from uniflex.core import modules
from uniflex.core import events
from uniflex.core import transactions
from uniflex.core.timer import TimerEventSender

__author__ = "Piotr Gawlowicz"
__copyright__ = "Copyright (c) 2016, Technische Universität Berlin"
__version__ = "0.1.0"
__email__ = "{gawlowicz}@tkn.tu-berlin.de"


class PeriodicEvaluationTimeEvent(events.TimeEvent):
    def __init__(self):
        super().__init__()


class MyController(modules.ControlApplication):
    def __init__(self):
        super(MyController, self).__init__()
        self.log = logging.getLogger('MyController')
        self.running = False

        self.timeInterval = 10
        self.timer = TimerEventSender(self, PeriodicEvaluationTimeEvent)
        self.timer.start(self.timeInterval)

        self.packetLossEventsEnabled = False

    @modules.on_start()
    def my_start_function(self):
        print("start control app")
        self.running = True

    @modules.on_exit()
    def my_stop_function(self):
        print("stop control app")
        self.running = False

    @modules.on_event(events.NewNodeEvent)
    def add_node(self, event):
        node = event.node

        self.log.info("Added new node: {}, Local: {}"
                      .format(node.uuid, node.local))
        self._add_node(node)

        for dev in node.get_devices():
            print("Dev: ", dev.name)
            print(dev)

        for m in node.get_modules():
            print("Module: ", m.name)
            print(m)

        for app in node.get_control_applications():
            print("App: ", app.name)
            print(app)

    @modules.on_event(events.NodeExitEvent)
    @modules.on_event(events.NodeLostEvent)
    def remove_node(self, event):
        self.log.info("Node lost".format())
        node = event.node
        reason = event.reason
        if self._remove_node(node):
            self.log.info("Node: {}, Local: {} removed reason: {}"
                          .format(node.uuid, node.local, reason))

    def default_cb(self, data):
        node = data.node
        devName = None
        if data.device:
            devName = data.device.name
        msg = data.msg
        print("Default Callback: "
              "Node: {}, Dev: {}, Data: {}"
              .format(node.hostname, devName, msg))

    @modules.on_event(PeriodicEvaluationTimeEvent)
    def periodic_evaluation(self, event):
        # go over collected samples, etc....
        # make some decisions, etc...
        print("Periodic Evaluation")
        print("My nodes: ", [node.hostname for node in self.get_nodes()])
        self.timer.start(self.timeInterval)

        if len(self.get_nodes()) == 0:
            return

        node = self.get_node(0)
        device = node.get_device(0)

        # node.is_access_locked()
        # device.is_access_locked()
        ## atomic lock (check if locked and if not lock)
        # device.lock_access()
        # device.unlock_access()

        transaction = transactions.Transaction()
        task1 = transactions.Task()
        task1.set_entities(device)
        task1.set_save_point_func(func=device.get_channel, args=["wlan0"])
        task1.set_function(func=device.set_channel, args=[12, "wlan0"])
        transaction.add_task(task1)

        task2 = transactions.Task()
        task2.set_entities(device)
        task2.set_save_point_value(args=[10, "wlan0"])
        task2.set_function(func=device.set_tx_power, args=[8, "wlan0"])
        transaction.add_task(task2)

        # rollback configuration change
        # if it breaks the connection with the controller and any node
        transaction.rollback_if_connection_lost(True, timeout=10)

        # commit is a blocking call that
        # runs three-phase-commit (3PC) protocol
        transaction.commit()

        transaction.is_executed()
        transaction.is_rolled_back()
        tstatus = transaction.get_status()
        print("Transaction status: ", tstatus)
