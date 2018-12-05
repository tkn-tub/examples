import logging
import datetime
import random

from sbi.radio_device.events import PacketLossEvent
from uniflex.core import modules
from uniflex.core import events
from uniflex.core.timer import TimerEventSender
from common import AveragedSpectrumScanSampleEvent
from common import ChangeWindowSizeEvent

from UniFlexGym.interfaces.uniflex_controller import UniFlexController

__author__ = "Piotr Gawlowicz, Sascha Rösler"
__copyright__ = "Copyright (c) 2016, Technische Universität Berlin"
__version__ = "0.1.0"
__email__ = "{gawlowicz}@tkn.tu-berlin.de, s.resler@campus.tu-berlin.de"


class PeriodicEvaluationTimeEvent(events.TimeEvent):
    def __init__(self):
        super().__init__()


class UniflexChannelController(modules.ControlApplication, UniFlexController):
    def __init__(self,**kwargs):
        super(UniflexChannelController, self).__init__()
        self.log = logging.getLogger('ChannelController')
        self.running = False

        self.timeInterval = 10
#        self.timer = TimerEventSender(self, PeriodicEvaluationTimeEvent)
#        self.timer.start(self.timeInterval)

        self.packetLossEventsEnabled = False
        self.channel = 1
        
        self.observationSpace = []
        self.lastObservation = []
        
#        if not "openAI_controller" in kwargs:
#            raise ValueError("There is no OpenAI gym controller specified. Can not #find \"" + "openAI_controller" + "\" as kwargs in the config file.")
#        else:
#            __import__(kwargs["openAI_controller"], globals(), locals(), [], 0)
#            splits = kwargs["openAI_controller"].split('.')
#            class_name = splits[-1]
#            self.openAI_controller = class_name(self, kwargs)

    @modules.on_start()
    def my_start_function(self):
        print("start control app")
        self.running = True
#        self.openAI_controller.run()

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

        #device = node.get_device(0)
        #device.set_tx_power(15, "wlan0")
        #device.set_channel(random.randint(1, 11), "wlan0")
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

    def _get_device_by_uuids(self, node_uuid, dev_uuid):
        nodes = self.get_nodes()
        myNodes = [x for x in nodes if x.uuid == node_uuid]
        if(len(myNodes) is not 1):
            return None
        node = myNodes[0]
        devices = node.get_devices()
        myDevices = [x for x in devices if x.uuid == dev_uuid]
        if(len(myDevices) is not 1):
            return None
        return myDevices[0]

    def scheduled_get_channel_cb(self, data):
        node = data.node
        msg = data.msg
        dev = node.get_device(0)
        print("Scheduled get_channel; Power in "
              "Node: {}, Dev: {}, was set to: {}"
              .format(node.hostname, dev.name, msg))

    def set_channel(self, node_uuid, dev_uuid, ifaceName, channel_number, channel_width):
        device = self._get_device_by_uuids(node_uuid, dev_uuid)
        if device is None:
            return False
        if channel_width is not None:
            device.blocking(False).set_channel(channel_number, ifaceName, channel_width= channel_width)
        else:
            device.blocking(False).set_channel(channel_number, ifaceName)
        return True

    '''
    Returns a list of the bandwidth of all transmitted data from one
    controlled device to a client. The data is structured as follows:
    {
        'MAC_of_client1' : {
            'mac' : 'MAC_of_client1',
            'bandwidth': bandwidth to the client,
            'node': {
                'hostname': 'hostname of my AP node',
                'uuid': 'uuid of my AP node'
            },
            'device': {
                'name': 'device name of the AP's physical interface',
                'uuid': 'uuid of the device',
            },
            'interface': 'name of the interface'
        }
    }
    Notice: new devices have bandwidth 0!
    '''
    def get_bandwidth(self):
        bandwidth = {}
        for node in self.get_nodes():
            for device  in node.get_devices():
                if type(device.my_control_flow) is not list:
                    device.my_control_flow = []
                
                for flow in device.my_control_flow:
                    flow['old'] = True
                
                for interface in device.get_interfaces():
                    infos = device.get_info_of_connected_devices(interface)
                    
                    for mac in infos:
                        values = infos[mac]
                        newTxBytes = int(values['tx bytes'][0])
                        
                        flow =  [d for d in device.my_control_flow if d['mac address'] == mac]
                        if len(flow) > 0:
                            flow = flow[0]
                            dif = datetime.datetime.now() - flow['last update']
                            bandwidth[mac] = {
                                'bandwidth':(newTxBytes - flow['tx bytes'] ) / (dif.total_seconds() + dif.microseconds / 1000000.0),
                                'node': {'hostname': node.hostname, 'uuid': node.uuid},
                                'device': {'name': device.name, 'uuid': device.uuid},
                                'interface': interface}
                            flow['tx bytes'] = newTxBytes
                            flow['last update'] = datetime.datetime.now()
                            flow['old'] = False
                        else :
                            device.my_control_flow.append({'mac address' : mac, 'tx bytes' : newTxBytes, 'last update' : datetime.datetime.now(), 'old' : False})
                            bandwidth[mac] = {
                                'mac' : mac,
                                'bandwidth': 0,
                                'node': {'hostname': node.hostname, 'uuid': node.uuid},
                                'device': {'name': device.name, 'uuid': device.uuid},
                                'interface': interface}
                
                for flow in device.my_control_flow:
                    if flow['old']:
                        device.my_control_flow.remove(flow)
        return bandwidth

    '''
    Returns a data structure of all available interfaces in the system
    It is structured as follows:
    {
        'uuid_of_node_1': {
            'hostname' : 'hostname of node1',
            'uuid' : 'uuid of node1',
            'devices' : {
                'name' : 'name of device1',
                'uuid' : 'uuid of device1',
                'interfaces' : [
                    'name of iface1', 'name of iface2'
                ]
            },
            ...
        },
        ...
    }
    '''
    def get_interfaces(self):
        interfaces = {}
        for node in self.get_nodes():
            nodeinfo = {'hostname': node.hostname, 'uuid': node.uuid}
            devices = {}
            for device  in node.get_devices():
                devinfo = {'name': device.name, 'uuid': device.uuid}
                interfaces_tmp = []
                for interface in device.get_interfaces():
                    interfaces_tmp.append(interface)
                devinfo['interfaces'] = interfaces_tmp
                devices[device.uuid] = devinfo
            nodeinfo['devices'] = devices
            interfaces[node.uuid] = nodeinfo
        return interfaces

    '''
    Collects and returns a list of the channel to interface mapping
    [
        {'channel number' : 'number of the channel',
        'channel width' : 'width of the channel',
        'node': {
            'hostname': 'hostname of my AP node',
            'uuid': 'uuid of my AP node'
        },
        'device': {
            'name': 'device name of the AP's physical interface',
            'uuid': 'uuid of the device',
        },
        'interface': 'name of the interface'
    ]
    '''
    def get_channels(self):
        channel_mapping = []
        for node in self.get_nodes():
            for device  in node.get_devices():
                for interface in device.get_interfaces():
                    chnum = device.get_channel(interface)
                    chw = device.get_channel_width(interface)
                    
                    channel_mapping.append({
                        'channel number' : chnum,
                        'channel width' : chw,
                        'device' : {'name': device.name, 'uuid': device.uuid},
                        'node' : {'hostname': node.hostname, 'uuid': node.uuid},
                        'interface' : interface})
        return channel_mapping

    def simulate_flows(self):
        flows = []
        for node in self.get_nodes():
            for device  in node.get_devices():
                for interface in device.get_interfaces():
                    chnum = device.get_channel(interface)
                    chw = device.get_channel_width(interface)
                    infos = device.get_info_of_connected_devices(interface)
                    
                    for mac in infos:
                        flows.append({'mac address' : mac, 'channel number' : chnum, 'channel width' : chw, 'iface': interface})
                
        for node in self.get_nodes():
            for device  in node.get_devices():
                for interface in device.get_interfaces():
                    device.set_packet_counter(flows, interface)

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
        
        ifaces = self.get_interfaces()
        node_uuid = list(ifaces.keys())[0]
        dev_uuid  = list(ifaces[node_uuid]['devices'].keys())[0]
        ifaceName = ifaces[node_uuid]['devices'][dev_uuid]['interfaces'][0]
        
        print(self.get_channels())
        self.simulate_flows()
        print(self.get_bandwidth())
        
        for node in self.get_nodes():
            for device  in node.get_devices():
                for interface in device.get_interfaces():
                    self.set_channel(node.uuid, device.uuid, interface, self.channel, None)
                    self.channel += 1
                    if self.channel > 13:
                        self.channel = 1
    
    '''
    OpenAI Gym Uniflex env API
    '''
    
    
    def reset(self):
        self.observationSpace = self._create_client_list()
        self.actionSpace = self._create_interface_list()
        
        interfaces = self.get_interfaces()
        
        # set a start channel for each interface:
        channel = 1
        for node in interfaces:
            for device in node['devices']:
                for iface in device['interfaces']:
                    self.set_channel(
                        node['uuid'], device['uuid'], iface, channel, None)
                    channel += 5
                    if channel > 12:
                        channel = 1
        # clear bandwidth counter
        self.get_bandwidth()
        return
    
    def execute_action(self, action):
        for index, actionStep in action:
            interface = self.actionSpace[index]
            self.set_channel(interface['node'], interface['device'], interface['iface'], actionStep, None)
        return
    
    def render():
        return
    
    def get_observationSpace(self):
        return
    
    def get_actionSpace(self):
        return
    
    def get_observation(self):
        observation  = []
        bandwidth = self.get_bandwidth()
        bandwidth = sorted(bandwidth, key=lambda k: k['mac'])
        for client in self.observationSpace:
            bandwidth = self. _get_bandwidth_by_client( bandwidth, client)
            if bandwidth in None:
                bandwidth = 0
            observation.append(bandwidth)
        
        self.lastObservation = observation
        return observation
    
    # game over if there is a new interface
    def get_gameOver(self):
        clients = self._create_client_list()
        return len(set(clients).symmetric_difference(set(self.observationSpace))) == 0
    
    def get_reward(self):
        if len(self.lastObservation) > 0:
            return reduce(lambda x, y: x^2 + y, self.lastObservation)
        return 0
    
    
    
    def _get_bandwidth_by_client(self, bandwidthList, clientData):
        for client in bandwidthList:
            if (client['mac'] is clientData['mac']) and (client['node'] is clientData['node']) and (client['device'] is clientData['device']) and (client['iface'] is clientData['iface']):
                return client['bandwidth']
        return None
    
    def _create_client_list(self):
        clientList = []
        clients = self.get_bandwidth()
        for client in clients:
            clientList.append({'mac': client['mac'], 'node': client['node']['uuid'],
                'device': client['device']['uuid'], 'iface': client['interface']})
        clients = sorted(clients, key=lambda k: k['mac'])
        return clientList
    
    def _create_interface_list(self):
        interfaceList = []
        interfaces = self.get_interfaces()
        for node in interfaces:
            for device in node['devices']:
                for iface in device['interfaces']:
                    interfaceList.append({'node': node['uuid'], 'device': device['uuid'], 'iface': iface})
        return interfaceList
    
    
    
    
        '''
        print(self.get_bandwidth())
        
        print(self.get_nodes())
        for node in self.get_nodes():
            print(node.get_devices())
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
