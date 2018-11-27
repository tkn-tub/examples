#!/usr/bin/env python3

from UniFlexGym.interfaces.uniflex_controller import UniFlexController
#import os,sys,inspect
#current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#parent_dir = os.path.dirname(current_dir)
#sys.path.insert(0, parent_dir) 
from channel_controller import UniflexChannelController
from functools import reduce

import gym

class Controller(UniFlexController):
    def __init__(self, **kwargs):
        super()
        self.channel_controller = UniflexChannelController()
        self.observationSpace = []
        self.lastObservation = []
        return
    
    def reset(self):
        self.observationSpace = self._create_client_list()
        self.actionSpace = self._create_interface_list()
        
        interfaces = self.channel_controller.get_interfaces()
        
        # set a start channel for each interface:
        channel = 1
        for node in interfaces:
            for device in node['devices']:
                for iface in device['interfaces']:
                    self.channel_controller.set_channel(
                        node['uuid'], device['uuid'], iface, channel, None)
                    channel += 5
                    if channel > 12:
                        channel = 1
        # clear bandwidth counter
        self.channel_controller.get_bandwidth()
        return
    
    def execute_action(self, action):
        for index, actionStep in action:
            interface = self.actionSpace[index]
            self.channel_controller.set_channel(interface['node'], interface['device'], interface['iface'], actionStep, None)
        return
    
    def render():
        return
    
    def get_observationSpace(self):
        return
    
    def get_actionSpace(self):
        return
    
    def get_observation(self):
        observation  = []
        bandwidth = self.channel_controller.get_bandwidth()
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
        clients = self.channel_controller.get_bandwidth()
        for client in clients:
            clientList.append({'mac': client['mac'], 'node': client['node']['uuid'],
                'device': client['device']['uuid'], 'iface': client['interface']})
        clients = sorted(clients, key=lambda k: k['mac'])
        return clientList
    
    def _create_interface_list(self):
        interfaceList = []
        interfaces = self.channel_controller.get_interfaces()
        for node in interfaces:
            for device in node['devices']:
                for iface in device['interfaces']:
                    interfaceList.append({'node': node['uuid'], 'device': device['uuid'], 'iface': iface})
        return interfaceList
