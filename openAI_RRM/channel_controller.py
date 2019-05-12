import logging
import datetime
import random
import numpy
import sys
from math import *

from functools import reduce

from sbi.radio_device.events import PacketLossEvent
from uniflex.core import modules
from uniflex.core import events
from uniflex.core.timer import TimerEventSender
from common import AveragedSpectrumScanSampleEvent
from common import ChangeWindowSizeEvent

from gym import spaces

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
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        self.running = False

        self.timeInterval = 10

        self.packetLossEventsEnabled = False
        self.channel = 1
        self.availableChannels = []
        self.observationSpace = []
        self.registeredClients = self._create_client_list()
        self.lastObservation = []
        self.actionSet = []
        self.simulation = False
        
        if 'availableChannels' in kwargs:
            self.availableChannels = kwargs['availableChannels']
        
        if 'simulation' in kwargs:
            self.simulation = kwargs['simulation']

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
    
    '''
        Channel mapping controller
    '''
    def set_channel(self, node_uuid, dev_uuid, ifaceName, channel_number, channel_width):
        '''
            Set one channel to one AP
            :param node_uuid: UUID of AP node
            :param dev_uuid: UUID of AP device
            :param ifaceName: Name of AP interface
            :param channel_number: Number of new channel
            :param channel_width: Bandwidth of new channel
        '''
        device = self._get_device_by_uuids(node_uuid, dev_uuid)
        if device is None:
            return False
        if channel_width is not None:
            device.blocking(False).set_channel(channel_number, ifaceName, channel_width= channel_width, control_socket_path='/var/run/hostapd')
        else:
            device.blocking(False).set_channel(channel_number, ifaceName, control_socket_path='/var/run/hostapd')
        return True

    def get_bandwidth(self):
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

    def get_interfaces(self):
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

    def get_channels(self):
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
        '''
            Simulate packet counters on simulated APs 
        '''
        
        flows = []
        
        #collect state(channels and bandwidth) of all devices
        for node in self.get_nodes():
            for device  in node.get_devices():
                for interface in device.get_interfaces():
                    chnum = device.get_channel(interface)
                    chw = device.get_channel_width(interface)
                    infos = device.get_info_of_connected_devices(interface)
                    mac = device.get_address()
                    
                    flows.append({'mac address' : mac, 'channel number' : chnum, 'channel width' : chw, 'iface': interface})
        
        # simulate packet counter on AP modules
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
        self.reset()
        self.execute_action([1])
        print(self.get_observation())
    
    
    
    '''
    OpenAI Gym Uniflex env API
    '''
    
    
    def reset(self):
        self.registeredClients = self._create_client_list()
        self.observationSpace = self.get_observationSpace()
        self.actionSpace = self.get_actionSpace()
        self.actionSet = []
        
        interfaces = self.get_interfaces()
        
        # set a start channel for each interface:
        channel = 1
        for nodeUuid, node in interfaces.items():
            for devUuid, device in node['devices'].items():
                for iface in device['interfaces']:
                    self.set_channel(
                        node['uuid'], device['uuid'], iface, channel, None)
                    channel += 5
                    if channel > 12:
                        channel = 1
        # clear bandwidth counter
        if(self.simulation):
            self.simulate_flows()
        self.get_bandwidth()
        return
    
    def execute_action(self, action):
        '''
            Map scalar action to channel vector
            channel value = (action/numberOfChannels^AP_id) mod numberOfChannels
        '''
        for index, interface in enumerate(self._create_interface_list()):
            ifaceaction = int(action / (pow(len(self.availableChannels),index)))
            ifaceaction = ifaceaction % len(self.availableChannels)
            self.set_channel(interface['node'], interface['device'], interface['iface'],
                                self.availableChannels[ifaceaction], None)
        return
    
    def render():
        return
    
    def get_observationSpace(self):
        '''
            Returns observation space for open AI gym
            result is a MultiDiscrete vector space
            each component has the number of available channels. Is the same value for all entries
        '''
        maxValues = [len(self.availableChannels) for i in self._create_interface_list()]
        #return spaces.Box(low=0, high=numChannels, shape=(len(self._create_interface_list()),0), dtype=numpy.float32)
        return spaces.MultiDiscrete(maxValues)
        #spaces.Box(low=0, high=10000000, shape=(len(self.observationSpace),), dtype=numpy.float32)
    
    def get_actionSpace(self):
        '''
            Returns action space for open AI gym
            result is a Discrete scalar space
            dimension is NumberOfChannels^NumberOfAPs
        '''
        interfaceList = self._create_interface_list();
        if(len(interfaceList) > 0):
            self.log.info("UUIDs of the action space")
        for key, interface in enumerate(interfaceList):
            self.log.info(str(key) + ":" + interface['device'])
        if len(interfaceList) == 0:
            return spaces.Discrete(0)
        return spaces.Discrete(pow(len(self.availableChannels), len(interfaceList)))
    
    def get_observation(self):
        '''
            Returns vector with state (channel) of each AP
        '''
        channels = self.get_channels()
        observation = list(map(lambda x: x['channel number'], channels))
        return observation
    
    # game over if there is a new interface
    def get_gameOver(self):
        '''
            Test if topology changes
            Bases on information, which client is registered at which AP
        '''
        clients = self._create_client_list()
        clientHash = [i['mac'] + i['node'] + i['device'] + i['iface'] for i in clients]
        observationSpaceHash = [i['mac'] + i['node'] + i['device'] + i['iface'] for i in self.registeredClients]
        return not len(set(clientHash).symmetric_difference(set(observationSpaceHash))) == 0
    
    def get_reward(self):
        '''
            Calculate reward for the current state
            reward = sum (sqrt(throughput of client))
        '''
        # for simulation
        if(self.simulation):
            self.simulate_flows()
        
        bandwidthList = self.get_bandwidth()
        #bandwidth = sorted(bandwidth, key=lambda k: k['mac'])
        reward = 0
        for key in bandwidthList:
            item = bandwidthList[key]
            if item['bandwidth'] < 0:
                print("Bandwidth has invalid value: " + str(item['bandwidth']))
                print(bandwidthList)
                continue
            reward += sqrt(item['bandwidth'])
        return reward
    
    
    
    def _get_bandwidth_by_client(self, bandwidthList, clientData):
        '''
            extracts bandwidth of client from bandwidth list
            :param bandwidthList: List of all clients, the AP they are associated with and their bandwidth
            :param clientData: data of the client. 
        '''
        for mac, client in bandwidthList.items():
            if (mac == clientData['mac']) and (client['node']['uuid'] == clientData['node']) and (client['device']['uuid'] == clientData['device']) and (client['interface'] == clientData['iface']):
                return client['bandwidth']
        return None
    
    def _create_client_list(self):
        '''
            create linear client list
            result is list of dictionarys with attribute: mac, node, device, iface
        '''
        clientList = []
        clients = self.get_bandwidth()
        for mac, client in clients.items():
            clientList.append({'mac': mac, 'node': client['node']['uuid'],
                'device': client['device']['uuid'], 'iface': client['interface']})
        clientList = sorted(clientList, key=lambda k: k['mac'])
        return clientList
    
    def _create_interface_list(self):
        '''
            create linear ap list
            result is list of dictionarys with attribute: node, device, iface
        '''
        interfaceList = []
        interfaces = self.get_interfaces()
        for nodeUuid, node in interfaces.items():
            for devUuid, device in node['devices'].items():
                for iface in device['interfaces']:
                    interfaceList.append({'node': node['uuid'], 'device': device['uuid'], 'iface': iface})
        return interfaceList
