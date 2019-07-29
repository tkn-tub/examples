#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym
import UniFlexGym
#import tensorflow as tf
#import tensorflow.contrib.slim as slim
import numpy as np
#from tensorflow import keras
import argparse
import logging
import time
import csv
import os
from math import *


parser = argparse.ArgumentParser(description='Uniflex reader')
parser.add_argument('--config', help='path to the uniflex config file', default=None)
parser.add_argument('--output', help='path to a csv file for agent output data', default=None)
parser.add_argument('--plot', help='activate plotting', default=None)
parser.add_argument('--steptime', help='interval between two steps', default=1)
parser.add_argument('--steps', help='number of steps in this execution. If not set, the agents runs infinitly long', default=None)

args = parser.parse_args()
if not args.config:
    print("No config file specified!")
    os._exit(1)
if not args.output:
    print("No output file specified! - Skip data")

if args.plot:
    import matplotlib.pyplot as plt

ac_space = []

def map_action(mappedAction):
    action = np.zeros(len(ac_space.nvec))
    for index in range(len(ac_space.nvec)):
        # filter action by the index
        ifaceaction = int(mappedAction / (pow(ac_space.nvec[0] ,index)))
        ifaceaction = ifaceaction % ac_space.nvec[0]
        action[index] = ifaceaction
    return action


#create uniflex environment
env = gym.make('uniflex-v0')
#env.configure()
env.start_controller(steptime=float(args.steptime), config=args.config)

numChannels = 2
episode = 1

while True:
    run = 0
    
    state = env.reset()
    n = 0
    ac_space = env.action_space
    ob_space = env.observation_space
    print("Observation space: ", ob_space,  ob_space.dtype)
    print("Action space: ", ac_space, ac_space.nvec)

    a_size = int(pow(ac_space.nvec[0], ac_space.nvec.shape[0]))
    
    avg = []
    num = []
    maxreward = 1
    lastreward = 0
    lastaction = 0
    
    done = False
    
    if a_size == 0:
        print("there is no vaild AP - sleep 10 seconds")
        time.sleep(2)
        continue
    
    aps = int(log(a_size, numChannels))
    
    for i in range(a_size):
        avg.append(0)
        num.append(0)
    
    while not done:
        # generate random values
        randval = []
        for i in range(a_size):
            randval.append(np.random.normal(avg[i]/maxreward, 1/(pow(num[i],2) + 1), 1))
        
        # take index of highest value
        action = np.argmax(randval)
        
        #execute step
        actionVector = map_action(action)
        next_state, reward, done, _ = env.step(actionVector)
        
        #hysteresis
        if action != lastaction and abs(reward - lastreward) < 0.1:
            reward = reward * 0.75
        lastaction = action
        lastreward = reward
        
        # add reward for further execution
        avg[action] = (avg[action] * num[action] + reward) / (num[action] + 2)
        num[action] += 1
        
        maxreward = np.maximum(maxreward, reward)
        
        # statistics
        if args.output:
            with open(args.output, 'a') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow([reward, action, episode])
            csvFile.close()
        
        print ("Reward: " + str(reward))
        print ("GameOver: " + str(done))
        print ("Next Channels: " + str(next_state))
        print ("Channel selection:" + str(action))
        print ("Average:" + str(avg))
        print ("next step")
        
        if args.plot:
            plt.subplot(211)
            plt.plot(run, reward, 'bo')                 # Additional point
            plt.ylabel('reward')
            plt.subplot(212)
            plt.plot(run, action, 'bo')                 # Additional point
            plt.ylabel('action')
            plt.xlabel('step')
            plt.pause(0.05)
        
        run += 1
        
        if args.steps and int(args.steps) <= run:
            os._exit(1)
        
    episode += 1
