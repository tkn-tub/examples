import logging
import datetime
import random
from uniflex.core import modules
from uniflex.core import events
from uniflex.timer import TimerEventSender


__author__ = "Anatolij Zubow, Piotr Gawlowicz"
__copyright__ = "Copyright (c) 2016, Technische Universität Berlin"
__version__ = "0.1.0"
__email__ = "{zubow,gawlowicz}@tkn.tu-berlin.de"


class PeriodicEvaluationTimeEvent(events.TimeEvent):
    def __init__(self):
        super().__init__()


class MyController(modules.ControlApplication):
    def __init__(self):
        super(MyController, self).__init__()
        self.log = logging.getLogger('MyController')
        self.running = False
        self.nodes = []

        self.timeInterval = 10
        self.timer = TimerEventSender(self, PeriodicEvaluationTimeEvent)
        self.timer.start(self.timeInterval)

    @modules.on_start()
    def my_start_function(self):
        print("start control app")
        self.running = True

        node = self.localNode
        self.log.info("My local node: {}, Local: {}"
                      .format(node.hostname, node.local))

        for dev in node.get_devices():
            print("Dev: ", dev.name)

        for m in node.get_modules():
            print("Module: ", m.name)

        for apps in node.get_control_applications():
            print("App: ", m.name)

        device = node.get_device(0)
        ifaces = device.radio.get_interfaces()
        print(ifaces)
        iface0 = ifaces[0]
        print(device.radio.get_interface_info(iface0))

        newIface = "wlan10"
        device.radio.add_interface(newIface, 'managed')
        ifaces = device.radio.get_interfaces()
        print(ifaces)

        device.radio.set_interface_up(newIface)
        # print(device.radio.is_interface_up(newIface))
        # device.radio.set_interface_down(newIface)
        # print(device.radio.is_interface_up(newIface))

        device.radio.del_interface(newIface)
        ifaces = device.radio.get_interfaces()
        print(ifaces)

    @modules.on_exit()
    def my_stop_function(self):
        print("stop control app")
        self.running = False

    @modules.on_event(PeriodicEvaluationTimeEvent)
    def periodic_evaluation(self, event):
        # go over collected samples, etc....
        # make some decisions, etc...
        print("Periodic Evaluation")

        node = self.localNode
        device = node.get_device(0)

        self.log.info("My local node: {}, Local: {}"
                      .format(node.hostname, node.local))
        self.timer.start(self.timeInterval)

        # execute non-blocking function immediately
        device.radio.set_tx_power(random.randint(1, 20))

        # execute non-blocking function immediately, with specific callback
        device.radio.get_tx_power()

        newChannel = random.randint(1, 11)
        device.radio.set_channel(channel=newChannel)

        # execute blocking function immediately
        result = device.radio.get_channel()
        print("{} Channel is: {}".format(datetime.datetime.now(), result))
