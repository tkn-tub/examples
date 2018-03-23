import logging
import datetime
import random
import wishful_upis as upis
from uniflex.core import modules
from uniflex.core import events
from uniflex.core.timer import TimerEventSender
from common import AveragedSpectrumScanSampleEvent
from common import StartMyFilterEvent
from common import StopMyFilterEvent

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
        self.nodes = []

        self.timeInterval = 10
        self.timer = TimerEventSender(self, PeriodicEvaluationTimeEvent)
        self.timer.start(self.timeInterval)

        self.myFilterRunning = False
        self.packetLossEventsEnabled = False

    @modules.on_start()
    def my_start_function(self):
        print("start control app")
        self.running = True

        node = self.localNode
        self.log.info("My local node: {}, Local: {}"
                      .format(node.hostname, node.local))

        for dev in node.get_devices():
            print("Dev: ", dev.name)
            print(dev)

        for m in node.get_modules():
            print("Module: ", m.name)
            print(m)

        for app in node.get_control_applications():
            print("App: ", app.name)
            print(app)

        device = node.get_device(0)
        device.radio.set_tx_power(15, 'ath0')
        device.radio.set_channel(random.randint(1, 11), 'ath0')
        device.enable_event(upis.radio.PacketLossEvent)
        self.packetLossEventsEnabled = True
        device.start_service(
            upis.radio.SpectralScanService(rate=1000, f_range=[2200, 2500]))

    @modules.on_exit()
    def my_stop_function(self):
        print("stop control app")
        self.running = False

    @modules.on_event(upis.radio.PacketLossEvent)
    def serve_packet_loss_event(self, event):
        node = event.node
        device = event.device
        self.log.info("Packet loss in node {}, dev: {}"
                      .format(node.hostname, device.name))

    @modules.on_event(AveragedSpectrumScanSampleEvent)
    def serve_spectral_scan_sample(self, event):
        avgSample = event.avg
        self.log.info("Averaged Spectral Scan Sample: {}"
                      .format(avgSample))

    def default_cb(self, data):
        node = data.node
        devName = None
        if data.device:
            devName = data.device.name
        msg = data.msg
        print("Default Callback: "
              "Node: {}, Dev: {}, Data: {}"
              .format(node.hostname, devName, msg))

    def get_power_cb(self, data):
        node = data.node
        dev = data.device
        msg = data.msg
        print("Power in "
              "Node: {}, Dev: {}, was set to: {}"
              .format(node.hostname, dev.name, msg))

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

        if self.packetLossEventsEnabled:
            device.disable_event(upis.radio.PacketLossEvent)
            self.packetLossEventsEnabled = False
        else:
            device.enable_event(upis.radio.PacketLossEvent)
            self.packetLossEventsEnabled = True

        if self.myFilterRunning:
            self.send_event(StopMyFilterEvent())
            self.myFilterRunning = False
        else:
            self.send_event(StartMyFilterEvent())
            self.myFilterRunning = True

        # execute non-blocking function immediately
        device.blocking(False).radio.set_tx_power(random.randint(1, 20), 'ath0')

        # execute non-blocking function immediately, with specific callback
        device.callback(self.get_power_cb).radio.get_tx_power('ath0')

        # schedule non-blocking function delay
        device.delay(3).callback(self.default_cb).radio.get_tx_power("wlan0")

        # schedule non-blocking function exec time
        exec_time = datetime.datetime.now() + datetime.timedelta(seconds=3)
        newChannel = random.randint(1, 11)
        device.exec_time(exec_time).radio.set_channel(newChannel, 'ath0')

        # execute blocking function immediately
        result = device.radio.get_channel('ath0')
        print("{} Channel is: {}".format(datetime.datetime.now(), result))

        # exception handling, clean_per_flow_tx_power_table implementation
        # raises exception
        try:
            device.radio.clean_per_flow_tx_power_table("wlan0")
        except Exception as e:
            print("{} !!!Exception!!!: {}".format(
                datetime.datetime.now(), e))
