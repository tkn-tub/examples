import logging
import datetime
import random

from sbi.radio_device.events import PacketLossEvent
from uniflex.core import modules
from uniflex.core import events
from uniflex.core.timer import TimerEventSender
from common import AveragedSpectrumScanSampleEvent
from common import ChangeWindowSizeEvent

__author__ = "Piotr Gawlowicz, Sascha Rösler"
__copyright__ = "Copyright (c) 2016, Technische Universität Berlin"
__version__ = "0.1.0"
__email__ = "{gawlowicz}@tkn.tu-berlin.de, s.resler@campus.tu-berlin.de"


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
        self.channel = 1

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

        device = node.get_device(0)
        device.set_tx_power(15, "wlan0")
        device.set_channel(random.randint(1, 11), "wlan0")
        #device.packet_loss_monitor_start()
        #device.spectral_scan_start()
        # device.play_waveform()
        # TODO: is_implemented()

    @modules.on_event(events.NodeExitEvent)
    @modules.on_event(events.NodeLostEvent)
    def remove_node(self, event):
        self.log.info("Node lost".format())
        node = event.node
        reason = event.reason
        if self._remove_node(node):
            self.log.info("Node: {}, Local: {} removed reason: {}"
                          .format(node.uuid, node.local, reason))

    @modules.on_event(PacketLossEvent)
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
        msg = data.msg
        dev = node.get_device(0)
        print("Power in "
              "Node: {}, Dev: {}, was set to: {}"
              .format(node.hostname, dev.name, msg))

        newPwr = random.randint(1, 20)
        dev.blocking(False).set_tx_power(newPwr, "wlan0")
        print("Power in "
              "Node: {}, Dev: {}, was set to: {}"
              .format(node.hostname, dev.name, newPwr))

    def scheduled_get_channel_cb(self, data):
        node = data.node
        msg = data.msg
        dev = node.get_device(0)
        print("Scheduled get_channel; Power in "
              "Node: {}, Dev: {}, was set to: {}"
              .format(node.hostname, dev.name, msg))

    @modules.on_event(PeriodicEvaluationTimeEvent)
    def periodic_evaluation(self, event):
        # go over collected samples, etc....
        # make some decisions, etc...
        print("Periodic Evaluation")
        print("My nodes: ", [node.hostname for node in self.get_nodes()])
        self.timer.start(self.timeInterval)

        if len(self.get_nodes()) == 0:
            return

        flows = []
        for node in self.get_nodes():
            for device  in node.get_devices():
                device.spectral_scan_stop()
                chnum = device.get_channel("wlan0")
                chw = device.get_channel_width("wlan0")
                infos = device.get_info_of_connected_devices("wlan0")
                
                for mac in infos:
                    flows.append({'mac address' : mac, 'channel number' : chnum, 'channel width' : chw})
                
        for node in self.get_nodes():
            print ("work " + node.hostname)
            for device  in node.get_devices():
            
                if type(device.my_control_flow) is not list:
                    device.my_control_flow = []
                
                for flow in device.my_control_flow:
                    flow['old'] = True
                
                device.set_packet_counter(flows, "wlan0")
                chnum = device.get_channel("wlan0")
                chw = device.get_channel_width("wlan0")
                infos = device.get_info_of_connected_devices("wlan0")
                
                bandwidth = {}
                
                for mac in infos:
                    values = infos[mac]
                    newTxBytes = int(values['tx bytes'][0])
                    
                    flow =  [d for d in device.my_control_flow if d['mac address'] == mac]
                    if len(flow) > 0:
                        flow = flow[0]
                        dif = datetime.datetime.now() - flow['last update']
                        bandwidth[mac] = (newTxBytes - flow['tx bytes'] ) / (dif.total_seconds() + dif.microseconds / 1000000.0)
                        flow['tx bytes'] = newTxBytes
                        flow['last update'] = datetime.datetime.now()
                        flow['old'] = False
                    else :
                        device.my_control_flow.append({'mac address' : mac, 'tx bytes' : newTxBytes, 'last update' : datetime.datetime.now(), 'old' : False})
                
                for flow in device.my_control_flow:
                    if flow['old']:
                        device.my_control_flow.remove(flow)
                
                print ("device " + device.name + " operates on channel " + str(chnum) + " with a bandwidth of " + chw + " - change to channel " + str(self.channel))
                print(bandwidth)
                
                device.blocking(False).set_channel(self.channel, "wlan0")
                
                self.channel += 1
                if self.channel > 13:
                    self.channel = 1
        '''
        node = self.get_node(0)
        device = node.get_device(0)

        if device.is_packet_loss_monitor_running():
            device.packet_loss_monitor_stop()
            device.spectral_scan_stop()
        else:
            device.packet_loss_monitor_start()
            device.spectral_scan_start()

        avgFilterApp = None
        for app in node.get_control_applications():
            if app.name == "MyAvgFilter":
                avgFilterApp = app
                break

        if avgFilterApp.is_running():
            myValue = random.randint(1, 20)
            [nValue1, nValue2] = avgFilterApp.blocking(True).add_two(myValue)
            print("My value: {} + 2 = {}".format(myValue, nValue1))
            print("My value: {} * 2 = {}".format(myValue, nValue2))
            avgFilterApp.stop()

            newWindow = random.randint(10, 50)
            old = avgFilterApp.blocking(True).get_window_size()
            print("Old Window Size : {}".format(old))
            avgFilterApp.blocking(True).change_window_size_func(newWindow)
            nValue = avgFilterApp.blocking(True).get_window_size()
            print("New Window Size : {}".format(nValue))

        else:
            avgFilterApp.start()
            newWindow = random.randint(10, 50)
            event = ChangeWindowSizeEvent(newWindow)
            avgFilterApp.send_event(event)

        # execute non-blocking function immediately
        device.blocking(False).set_tx_power(random.randint(1, 20), "wlan0")

        # execute non-blocking function immediately, with specific callback
        device.callback(self.get_power_cb).get_tx_power("wlan0")

        # schedule non-blocking function delay
        device.delay(3).callback(self.default_cb).get_tx_power("wlan0")

        # schedule non-blocking function exec time
        exec_time = datetime.datetime.now() + datetime.timedelta(seconds=3)
        newChannel = random.randint(1, 11)
        device.exec_time(exec_time).set_channel(newChannel, "wlan0")

        # schedule execution of function multiple times
        start_date = datetime.datetime.now() + datetime.timedelta(seconds=2)
        interval = datetime.timedelta(seconds=1)
        repetitionNum = 3
        device.exec_time(start_date, interval, repetitionNum).callback(self.scheduled_get_channel_cb).get_channel("wlan0")

        # execute blocking function immediately
        result = device.get_channel("wlan0")
        print("{} Channel is: {}".format(datetime.datetime.now(), result))

        # exception handling, clean_per_flow_tx_power_table implementation
        # raises exception
        try:
            device.clean_per_flow_tx_power_table("wlan0")
        except Exception as e:
            print("{} !!!Exception!!!: {}".format(
                datetime.datetime.now(), e))
        '''
