#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##TODO: Insert path to neuronal network

import gym
import UniFlexGym
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from tensorflow import keras
import argparse
import logging
import time
import csv
import os
from math import *
from scipy.optimize import fsolve
from gym import spaces

sortedIndecies = []
ac_space = []
BANDWITH_ON_CHANNEL = 54e6
numChannels = 2
SORT_VALUES = True
topologies = [[1,2,1], [2,2,2], [1,0,1]]
N_test = int(1000)
aps = 3
maxclients = 100

def zeros(anz):
    return np.zeros(anz)

def ones(anz):
    return np.ones(anz)

def calculate_all_best_action(clients, aps):
    aps_sort = sorted(aps)
    #if string topology
    if aps_sort == [1,1,2]:
        #set ap in the middle to 1, all other to 0
        result = [zeros(3), ones(3)]
        result[0][aps.index(2)] = 1
        result[1][aps.index(2)] = 0
        return result
    
    # if island topology
    if aps_sort == [0,1,1]:
        #set on of the neighouring aps to 1, all other to 0
        result = []
        group = np.where(np.array(aps) == 1)
        for elem in group[0]:
            myresult = zeros(3)
            myresult[elem] = 1
            result.append(myresult)
            myresult = ones(3)
            myresult[elem] = 0
            result.append(myresult)
        return result
    
    #if all aps can hear all other
    if aps_sort == [2,2,2]:
        result = []
        #get ap with most clients
        clients_sort = sorted(clients)
        ap_most = np.where(clients == clients_sort[2])
        for elem in ap_most[0]:
            myresult = zeros(3)
            myresult[elem] = 1
            result.append(myresult)
            myresult = ones(3)
            myresult[elem] = 0
            result.append(myresult)
        return result
    # there is no topology
    print("notopo")
    return [zeros(3)]

#test data consist on a vector of observations and a label vector of all valid 
# action to this observations
def generate_testdata(number, dimension, maxclients, topoplogies, sortValues):
    data = []
    labels = []
    for i in range(number):
        for topology in topologies:
            index = np.array(range(dimension), dtype=np.int16)
            clients = np.random.randint(maxclients, size=dimension)
            mydata = np.vstack((clients, topology, index)).transpose()
            labeldata = mydata
            #if sortValues:
            #    labeldata = np.sort(labeldata.view('i4,i4'), order=['f0', 'f1'], axis=0).view(np.int)
            clients = labeldata[:,0]
            topology = labeldata[:,1].tolist()
            #mydata = np.reshape(mydata, [1, 2*dimension])
            mydata = np.delete(mydata, 2, axis=1)
            data.append(mydata)
            #labels.append(channelvectors_to_label(calculate_all_best_action(clients, topology)))
            labels.append(calculate_all_best_action(clients, topology))
    return [data, labels]

def normalize_state(state, ob_space, s_size):
    global sortedIndecies
    state = np.array(state)
    
    #sort states
    index = np.arange(state.shape[0])
    index = index.reshape((-1,1))
    state = np.concatenate((state, index), axis=1)
    #
    if SORT_VALUES:
        state = np.sort(state.view('i8,i8,i8'), order=['f0', 'f1'], axis=0).view(np.int)
    sortedIndecies = state[:,-1]
    state = np.delete(state, -1, axis=1)
    state = np.reshape(state, [1, s_size])
    obspacehigh = np.reshape(ob_space.high, [1, s_size])
    #state = state *2 / obspacehigh - 1
    state = state -1
    
    return state

def map_action(mappedAction):
    action = np.zeros(len(ac_space.nvec))
    for index in range(len(ac_space.nvec)):
        # filter action by the index
        ifaceaction = int(mappedAction / (pow(ac_space.nvec[0] ,index)))
        ifaceaction = ifaceaction % ac_space.nvec[0]
        #print("ifaceaction at " + str(index) + " is " + str(ifaceaction))
        #print("Find " + str(index) + "in sorted indecies" + str(sortedIndecies)+ "at" + str(np.where(sortedIndecies == index)))
        #action[np.where(sortedIndecies == index)[0]] = ifaceaction
        action[sortedIndecies[index]] = ifaceaction
    return action

def eval(clients):
    errorcounter_cli = 0
    errorcounter_ap  = 0
    counter = 0
    errorlog = ""

    for client in clients:
        ap = client['aps']
        state_cli = np.array([client['clients'], ap])
        #state_ap  = np.array([ap, client['clients']])
        
        state_cli = state_cli.transpose()
        #state_ap  = state_ap.transpose()
        
        state_cli_norm = normalize_state(state_cli.tolist(), ob_space, s_size)
        action = np.argmax(model.predict(state_cli_norm)[0])
        actionvector = map_action(action)
        
        #state_ap_norm = normalize_state(state_ap.tolist(), ob_space, s_size)
        #actionap = np.argmax(modelap.predict(state_ap_norm)[0])
        #actionvectorap = map_action(actionap)
        
        success_cli = False
        for tmp in client['valid']:
            tmpval = True
            for a, b in zip(actionvector, tmp):
                if a != b:
                    tmpval = False
                    break
            if tmpval:
                success_cli = True
                break
        
        #success_ap = False
        #for tmp in client['valid']:
        #    tmpval = True
        #    for a, b in zip(actionvectorap, tmp):
        #        if a != b:
        #            tmpval = False
        #            break
        #    if tmpval:
        #        success_ap = True
        #        break
        
        output = "[Cli, Ap]: Cli:" + str(client['clients']) + ", AP:" + str(ap) + ", Action:" +str(action) + ", Actionvector" + str(actionvector) + ", " + str(success_cli) + " SortedID: " +str(sortedIndecies)
        print(output)
        #print("[Ap, Cli]: Cli:" + str(client['clients']) + ", AP:" + str(ap) + ", Action:" +str(actionap) + ", Actionvector" + str(actionvectorap) + ", " + str(success_ap))
        counter += 1
        
        #if not success_ap:
        #    errorcounter_ap +=1
        
        if not success_cli:
            errorcounter_cli +=1
            errorlog += output +"\n"

    print("Errors in [Cli,Ap]:" + str(errorcounter_cli) + "/" + str(counter) + "(" + str(errorcounter_cli/counter*100) + "%)")
    #print("Errors in [Ap,Cli]:" + str(errorcounter_ap) + "/" + str(counter) + "(" + str(errorcounter_ap/counter) + "%)")
    print(errorlog)

def calculate_reward(clients_p_ap, action):
    reward = 0
    
    for ap in range(len(action)):
        channel = action[ap]
        
        #search num aps on same channel
        same_chan = 0
        for act in action:
            if act == channel:
                same_chan += 1
        
        ap_bandwidth = BANDWITH_ON_CHANNEL/ same_chan
        reward += clients_p_ap[ap] * sqrt(ap_bandwidth/clients_p_ap[ap])
    return reward

def get_best_reward(client, ap):
    state_cli = np.array([client, ap])
    #state_ap  = np.array([ap, client['clients']])
    
    state_cli = state_cli.transpose()
    #state_ap  = state_ap.transpose()
    
    state_cli_norm = normalize_state(state_cli.tolist(), ob_space, s_size)
    action = np.argmax(model.predict(state_cli_norm)[0])
    actionvector = map_action(action)
    
    reward = calculate_reward(client, actionvector)
    return reward

def eval_handover(client, new_clients):
    print("Current state:")
    ap = client['aps']
    state_cli = np.array([client['clients'], ap])
    
    state_cli = state_cli.transpose()
    state_cli_norm = normalize_state(state_cli.tolist(), ob_space, s_size)
    action = np.argmax(model.predict(state_cli_norm)[0])
    actionvector = map_action(action)
    
    success_cli = False
    for tmp in client['valid']:
        tmpval = True
        for a, b in zip(actionvector, tmp):
            if a != b:
                tmpval = False
                break
        if tmpval:
            success_cli = True
            break
    
    reward = get_best_reward(client['clients'], ap)
    
    print("Cli:" + str(client['clients']) + ", AP:" + str(ap) + ", Action:" +str(action) + ", Actionvector" + str(actionvector) + ", " + str(success_cli) + ", reward:" + str(reward))
    
    print("Handover simulation")
    for new_client in new_clients:
        ap = new_client['aps']
        state_cli = np.array([new_client['clients'], ap])
        
        state_cli = state_cli.transpose()
        state_cli_norm = normalize_state(state_cli.tolist(), ob_space, s_size)
        action = np.argmax(model.predict(state_cli_norm)[0])
        actionvector = map_action(action)
        reward = calculate_reward(new_client['clients'], actionvector)
        
        success_cli = False
        for tmp in new_client['valid']:
            tmpval = True
            for a, b in zip(actionvector, tmp):
                if a != b:
                    tmpval = False
                    break
            if tmpval:
                success_cli = True
                break
        
        print("Cli:" + str(new_client['clients']) + ", AP:" + str(ap) + ", Action:" +str(action) + ", Actionvector" + str(actionvector) + ", " + str(success_cli) + ", reward:" + str(reward))


ac_space = spaces.MultiDiscrete([2,2,2])
ob_space = spaces.Box(low=0, high=6, shape=(ac_space.nvec.shape[0],2), dtype=np.uint32)
trainingfile = "path-to-neuronal-network.train"

#generate random test data
[test_data, test_labels] = generate_testdata(N_test, aps, maxclients, topologies, SORT_VALUES)

clients = []
for elem, label in zip(test_data, test_labels):
    clients.append({'clients': elem[:,0], 'aps': elem[:,1], 'valid':label})


special = [{'clients': [10, 10, 1000], 'aps': [2,2,2], 'valid':[[1,1,0], [0,0,1]]},
            {'clients': [1000, 10, 10], 'aps': [1,2,1], 'valid':[[1,0,1], [0,1,0]]},
            {'clients': [1000, 4, 2], 'aps': [1,0,1], 'valid':[[1,0,0], [0,0,1],[1,1,0], [0,1,1]]}
            ]


print("Observation space: ", ob_space,  ob_space.dtype)
print("Action space: ", ac_space, ac_space.nvec)

tmps_size = ob_space.shape
s_size = tmps_size[0] * tmps_size[1]
#s_size = list(map(lambda x: x * ob_space.high, s_size))
a_size = pow(ac_space.nvec[0], ac_space.nvec.shape[0])

print("observation_space size:" + str(s_size))
print("Data: Trained Data of different settings with sorting agent. Experiment 3")

model = keras.Sequential()
model.add(keras.layers.Dense(s_size, input_shape=(s_size,), activation='relu'))
#model.add(keras.layers.Dense(5, activation='relu'))
model.add(keras.layers.Dense(a_size, activation='softmax'))
model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.load_weights(trainingfile)

print("\n3000 zufällige Szenarien:")
eval(clients)

print("\Spezielle Szenarien:")
eval(special)

