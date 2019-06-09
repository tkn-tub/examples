#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


def normalize_state(state, ob_space, s_size):
    state = np.reshape(state, [1, s_size])
    obspacehigh = np.reshape(ob_space.high, [1, s_size])
    state = state *2 / obspacehigh - 1
    return state

parser = argparse.ArgumentParser(description='Uniflex reader')
parser.add_argument('--config', help='path to the uniflex config file', default=None)
parser.add_argument('--output', help='path to a csv file for agent output data', default=None)
parser.add_argument('--plot', help='activate plotting', default=None)
parser.add_argument('--steptime', help='interval between two steps', default=1)
parser.add_argument('--steps', help='number of steps per episode. If not set, the agents runs infinitly long', default=None)
parser.add_argument('--episodes', help='number of episodes in this execution. If not set, the agents runs infinitly long', default=1)
parser.add_argument('--trainingfile', help='file to load and store training data', default=None)

args = parser.parse_args()
if not args.config:
    print("No config file specified!")
    os._exit(1)
if not args.output:
    print("No output file specified! - Skip data")
if not args.trainingfile:
    print("No training file specified! - Start with unlearned agent")
    
if args.plot:
    import matplotlib.pyplot as plt

#create uniflex environment, steptime is 10sec
env = gym.make('uniflex-v0')
#env.configure()
env.start_controller(steptime=float(args.steptime), config=args.config)

epsilon_max = 1.0               # exploration rate
epsilon_min = 0.01
#epsilon_decay = 0.99
epsilon_decay = 0.995

time_history = []
rew_history = []

numChannels = 2

while True:
    
    state = env.reset()
    
    n = 0
    ac_space = env.action_space
    ob_space = env.observation_space
    
    print("reset agent")
    print("Observation space: ", ob_space,  ob_space.dtype)
    print("Action space: ", ac_space, ac_space.n)

    tmps_size = ob_space.shape
    s_size = tmps_size[0] * tmps_size[1]
    #s_size = list(map(lambda x: x * ob_space.high, s_size))
    a_size = ac_space.n
    
    print(s_size)
    
    state = normalize_state(state, ob_space, s_size)
    
    model = keras.Sequential()
    model.add(keras.layers.Dense(s_size, input_shape=(s_size,), activation='relu'))
    model.add(keras.layers.Dense(5, activation='relu'))
    model.add(keras.layers.Dense(a_size, activation='softmax'))
    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    if args.trainingfile and not os.path.isfile(args.trainingfile):
        try:
            model.load_weights(args.trainingfile)
            print("Load model")
        except ValueError:
            print("Spaces does not match")
    
    print(state)
    try:
        state = np.reshape(state, [1, s_size])
    except ValueError:
        continue
    rewardsum = 0
    
    if a_size == 0:
        print("there is no vaild AP - sleep 2 seconds")
        time.sleep(2)
        continue
    
    episode = 1
    
    # Schleife für Episoden
    while True:
        print("start episode")
        
        run = 0
        runs = []
        rewards = []
        actions = []
        maxreward = 0.00001
        minreward = np.inf
        
        epsilon = epsilon_max
        epsilon_max *= 0.999
        epsilon_max = pow(epsilon_max, 3)
        done = False
        lastreward = 0
        lastaction = 0
        
        aps = int(log(a_size, numChannels))
        
        for i in range(0, aps):
            actions.append([])
        
        state = env.reset()
        state = normalize_state(state, ob_space, s_size)
        
        while not done:
            # Choose action
            if np.random.rand(1) < epsilon:
                action = np.random.randint(a_size)
            else:
                action = np.argmax(model.predict(state)[0])

            # Step
            next_state, reward, done, _ = env.step(action)
            
            minreward = min(reward, minreward)
            reward -= minreward
            
            maxreward = max(reward, maxreward)
            reward /= maxreward
            
            #set reward to 1.0 if it is first value
            if maxreward == 0.00001:
                reward = 1.0
            
            reward = pow(reward, 2)
            
            #hysteresis
            if action != lastaction and abs(reward - lastreward) < 0.1:
                reward *= 0.9
            lastaction = action
            lastreward = reward
            

            if done:
            #    print("episode: {}/{}, time: {}, rew: {}, eps: {:.2}"
            #          .format(e, total_episodes, time, rewardsum, epsilon))
                break

            
            next_state = normalize_state(next_state, ob_space, s_size)

            # Train
            target = reward
            if not done:
                target = (reward)# + 0.95 * np.amax(model.predict(next_state)[0]))
            
            print(target)
            
            target_f = model.predict(state)
            print(target_f)
            target_f[0][action] = target
            print(target_f)
            model.fit(state, target_f, epochs=1, verbose=0)

            state = next_state
            #rewardsum += reward
            if epsilon > epsilon_min: epsilon *= epsilon_decay
            
            rewards.append(reward)
            
            if args.trainingfile:
                model.save_weights(args.trainingfile)
            
            if args.output:
                with open(args.output, 'a') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow([reward, action, episode])
                csvFile.close()
            
            for ap in range(0, aps):
                ifaceaction = int(action / (pow(numChannels, ap)))
                ifaceaction = ifaceaction % numChannels
                actions[ap].append(ifaceaction)
            
            print ("Reward: " + str(reward))
            print ("GameOver: " + str(done))
            print ("State: " + str(state))
            print ("Channel selection:" + str(action))
            print ("Run: " + str(run) + ", Episode: " + str(episode))
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
            
            run += 1
            
            # next episode if enough steps, if enough episodes -> exit
            if args.steps and int(args.steps) <= run:
                if args.episodes and int(args.episodes) <= episode:
                    os._exit(1)
                else:
                    break
            
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
