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


#create uniflex environment, steptime is 10sec
env = gym.make('uniflex-v0')
#env.configure()
env.start_controller(steptime=float(args.steptime), config=args.config)

epsilon = 1.0               # exploration rate
epsilon_min = 0.01
#epsilon_decay = 0.99
epsilon_decay = 0.995

time_history = []
rew_history = []

numChannels = 2
episode = 0

while True:
    run = 0
    runs = []
    rewards = []
    actions = []
    
    state = env.reset()
    n = 0
    ac_space = env.action_space
    ob_space = env.observation_space
    print("Observation space: ", ob_space,  ob_space.dtype)
    print("Action space: ", ac_space, ac_space.n)

    a_size = int(ac_space.n)
    
    avg = []
    num = []
    maxreward = 1
    
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
            randval.append(np.random.normal(avg[i]/maxreward, 1/(num[i] + 1), 1))
        
        #take index of highest value
        action = np.argmax(randval)
        
        #execute step
        next_state, reward, done, _ = env.step(action)
        
        # add reward for further execution
        avg[action] = (avg[action] * num[action] + reward) / (num[action] + 2)
        num[action] += 1
        
        maxreward = np.maximum(maxreward, reward)
        
        # statistics
        rewards.append(reward)
        
        if args.output:
            with open(args.output, 'a') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow([reward, action])
            csvFile.close()
        
        for ap in range(0, aps):
            ifaceaction = int(action / (pow(numChannels, ap)))
            ifaceaction = ifaceaction % numChannels
            #actions[ap].append(ifaceaction)
        
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
            #for ap in range(0, aps):
            #    plt.plot(actions[ap])
            plt.plot(run, action, 'bo')                 # Additional point
            plt.ylabel('action')
            plt.xlabel('step')
            plt.pause(0.05)
        
        if args.steps and int(args.steps) < run:
            os._exit(1)
        
        run += 1
        
    episode += 1


'''
ob_space = env.observation_space
ac_space = env.action_space
print("Observation space: ", ob_space,  ob_space.dtype)
print("Action space: ", ac_space, ac_space.n)

s_size = ob_space.shape[0]
a_size = ac_space.n
model = keras.Sequential()
model.add(keras.layers.Dense(s_size, input_shape=(s_size,), activation='relu'))
model.add(keras.layers.Dense(a_size, activation='softmax'))
model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

total_episodes = 200
max_env_steps = 100
env._max_episode_steps = max_env_steps

epsilon = 1.0               # exploration rate
epsilon_min = 0.01
epsilon_decay = 0.999

time_history = []
rew_history = []

for e in range(total_episodes):

    state = env.reset()
    state = np.reshape(state, [1, s_size])
    rewardsum = 0
    for time in range(max_env_steps):
        # Choose action
        if np.random.rand(1) < epsilon:
            action = np.random.randint(a_size)
        else:
            action = np.argmax(model.predict(state)[0])
        
        # Step
        next_state, reward, done, _ = env.step(action)
        
        if done:
            print("episode: {}/{}, time: {}, rew: {}, eps: {:.2}"
                  .format(e, total_episodes, time, rewardsum, epsilon))
            break
        
        next_state = np.reshape(next_state, [1, s_size])
        
        # Train
        target = reward
        if not done:
            target = (reward + 0.95 * np.amax(model.predict(next_state)[0]))
        
        target_f = model.predict(state)
        target_f[0][action] = target
        model.fit(state, target_f, epochs=1, verbose=0)
        
        state = next_state
        rewardsum += reward
        if epsilon > epsilon_min: epsilon *= epsilon_decay
        
    time_history.append(time)
    rew_history.append(rewardsum)
'''
