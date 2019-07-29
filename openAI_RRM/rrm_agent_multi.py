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
import json
from math import *
from scipy.optimize import fsolve
import pickle
import datetime
from functools import reduce

AVGTIME_ONEVALUE_RAND = 2
RANDVALUE_FIRST_EPISODE = 0.9
REWARD_INIT = 0.00001
SORT_VALUES = True
SCENARIOS = 180
EPSILON_MAX_DECAY = 0.95
EPSILON_MIN = 0.01
ACTIVATE_OBSERVER = False

sortedIndecies = []
ac_space = []
currentScenario = 0

def normalize_state(state, ob_space, s_size):
    global sortedIndecies
    state = np.array(state)
    #sort states
    index = np.arange(state.shape[0])
    index = index.reshape((-1,1))
    state = np.concatenate((state, index), axis=1)
    #sort input and output if configured
    if SORT_VALUES:
        state = np.sort(state.view('i8,i8,i8'), order=['f0', 'f1'], axis=0).view(np.int)
    #print("state" + str(state))
    sortedIndecies = state[:,-1]
    #print(sortedIndecies)
    state = np.delete(state, -1, axis=1)
    
    state = np.reshape(state, [1, s_size])
    # obspacehigh = np.reshape(ob_space.high, [1, s_size])
    state = state - 1 #*2 / obspacehigh - 1
    
    return state

def guess_random_numbers_in_firstEpisode(a_size):
    return AVGTIME_ONEVALUE_RAND * a_size * SCENARIOS#**2

def guess_steps(a_size):
    stepidea = guess_random_numbers_in_firstEpisode(a_size) / RANDVALUE_FIRST_EPISODE
    #scale to multiple of scenario
    stepidea = int(stepidea / SCENARIOS) * SCENARIOS
    return stepidea

def guess_epsilon_decay(steps, a_size):
    func = lambda epsilon_decay: guess_random_numbers_in_firstEpisode(a_size) - (1-epsilon_decay**(steps + 1)) / (1 - epsilon_decay)
    return fsolve(func, 0.9999999999)[0]

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

def reset_rewards():
    global maxreward
    global minreward;
    for i in range(SCENARIOS):
        maxreward[i] = REWARD_INIT
        minreward[i] = np.inf
    return

def normalize_reward(reward, rewardpow, action):
    global maxreward
    global minreward;
    global lastreward;
    global currentScenario;
    
    orig = reward
    
    minreward[currentScenario] = min(reward, minreward[currentScenario])
    reward -= minreward[currentScenario]
    
    maxreward[currentScenario] = max(reward, maxreward[currentScenario])
    reward /= maxreward[currentScenario]
    
    print("reward:" + str(orig) + ", minreward:" + str(minreward[currentScenario]) + ", maxreward:" +str(maxreward[currentScenario]) + ", at scenario" + str(currentScenario))
    
    #set reward to 1.0 if it is first value
    if maxreward[currentScenario] == REWARD_INIT:
        reward = 1.0
    
    reward = pow(reward, rewardpow)
    
    #hysteresis
    if action != lastaction[currentScenario] and abs(reward - lastreward[currentScenario]) < 0.1:
        reward *= 0.9
    lastaction[currentScenario] = action
    lastreward[currentScenario] = reward
    
    return reward

lastreward = np.zeros(SCENARIOS)
minreward = np.zeros(SCENARIOS)
maxreward = np.zeros(SCENARIOS)
lastaction = np.zeros(SCENARIOS)


parser = argparse.ArgumentParser(description='Uniflex reader')
parser.add_argument('--config', help='path to the uniflex config file', default=None)
parser.add_argument('--output', help='path to a csv file for agent output data', default=None)
parser.add_argument('--plot', help='activate plotting', default=None)
parser.add_argument('--steptime', help='interval between two steps', default=1)
#parser.add_argument('--steps', help='number of steps per episode. If not set, the agents runs infinitly long', default=None)
parser.add_argument('--episodes', help='number of episodes in this execution. If not set, the agents runs infinitly long', default=None)
parser.add_argument('--startepisode', help='The episode we start with', default=1)
parser.add_argument('--trainingfile', help='file to load and store training data', default=None)
parser.add_argument('--cpus', help='Numbers of cpus for this process', default=1)

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

print("Start at episode " + str(args.startepisode))

#create uniflex environment, steptime is 10sec
env = gym.make('uniflex-v0')
#env.configure()
env.start_controller(steptime=float(args.steptime), config=args.config)

epsilon_max = 1.0               # exploration rate
#epsilon_decay = 0.99
#epsilon_decay = 0.995

time_history = []
rew_history = []

numChannels = 2

observerData = []
observerCounter = 0

while True:
    
    state = env.reset()
    currentScenario = 0
    
    n = 0
    ac_space = env.action_space
    ob_space = env.observation_space
    
    print("reset agent")
    print("Observation space: ", ob_space,  ob_space.dtype)
    print("Action space: ", ac_space, ac_space.nvec)

    tmps_size = ob_space.shape
    s_size = tmps_size[0] * tmps_size[1]
    #s_size = list(map(lambda x: x * ob_space.high, s_size))
    a_size = pow(ac_space.nvec[0], ac_space.nvec.shape[0])
    
    if a_size == 0:
        print("there is no vaild AP - sleep 2 seconds")
        time.sleep(2)
        continue
    
    print("observation_space size:" + str(s_size))
    
    state = normalize_state(state, ob_space, s_size)
    
    model = keras.Sequential()
    model.add(keras.layers.Dense(s_size, input_shape=(s_size,), activation='relu'))
    #model.add(keras.layers.Dense(5, activation='relu'))
    model.add(keras.layers.Dense(a_size, activation='softmax'))
    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    config = tf.ConfigProto()
    config.intra_op_parallelism_threads = int(args.cpus)
    config.inter_op_parallelism_threads = int(args.cpus)
    tf.Session(config=config)

    print("State (Observation) of System" + str(state))
    try:
        state = np.reshape(state, [1, s_size])
    except ValueError:
        continue
    rewardsum = 0
    
    steps = guess_steps(a_size)
    epsilon_decay = guess_epsilon_decay(steps, a_size)
    print("Initialize agent. Exploration rate is " + str(epsilon_decay) 
        + ", an episode has at most " + str(steps) + " steps")
    
    rewardpow = int(log(a_size, 2))
    
    episode = 1
    reset_rewards()
    
    if args.trainingfile and not os.path.isfile(args.trainingfile):
        try:
            model.load_weights(args.trainingfile)
            print("Load model")
        except ValueError:
            print("Spaces does not match")
        except tf.errors.NotFoundError:
            print("File not found. Skip loading")
        
        try:
            with open(args.trainingfile + '.var', 'r') as f:  # Python 3: open(..., 'wb')
                temp = json.loads(f.read())
                lastreward = np.array(temp['lastreward'])
                minreward  = np.array(temp['minreward'])
                maxreward  = np.array(temp['maxreward'])
                lastaction = np.array(temp['lastaction'])
                print("Load reward values of last run")
                print("lastreward: " + str(lastreward))
                print("minreward: " + str(minreward))
                print("maxreward: " + str(maxreward))
                print("lastaction: " + str(lastaction))
        except ValueError as e:
            print("File format is wrong" + str(e))
        except IOError as e:
            print("File not found. Skip loading" + str(e))
    
    
    while episode < int(args.startepisode):
        epsilon_max *= EPSILON_MAX_DECAY
        epsilon_max = max(epsilon_max, EPSILON_MIN)#max(pow(epsilon_max, 3), EPSILON_MIN)
        episode += 1
    
    # Schleife für Episoden
    while True:
        print("start episode")
        
        run = 0
        runs = []
        rewards = []
        actions = []
        
        epsilon = epsilon_max
        epsilon_max *= EPSILON_MAX_DECAY
        epsilon_max = max(epsilon_max, EPSILON_MIN)#max(pow(epsilon_max, 3), EPSILON_MIN)
        done = False
        
        aps = int(log(a_size, numChannels))
        
        #for i in range(0, aps):
        #    actions.append([])
        
        state = env.reset()
        state_orig = state
        currentScenario = 0
        state = normalize_state(state, ob_space, s_size)
        
        while not done:
            # Choose action
            ts = time.time()
            print("\nnew step at " + datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S,%f'))
            print ("Run: " + str(run) + ", Episode: " + str(episode) + ", Scenario: " + str(currentScenario))
            print("Observation:" + str(state_orig))
            
            if np.random.rand(1) < epsilon:
                action = np.random.randint(a_size)
            else:
                action = np.argmax(model.predict(state)[0])

            actionvector = map_action(action)
            
            print("Action:" +str(action) + ", Actionvector" + str(actionvector))
            
            # Step
            next_state, reward, done, _ = env.step(actionvector)
            
            reward = normalize_reward(reward, rewardpow, action)
            

            if done:
            #    print("episode: {}/{}, time: {}, rew: {}, eps: {:.2}"
            #          .format(e, total_episodes, time, rewardsum, epsilon))
                reset_rewards()
                print("setting changes")
                break
            
            state_orig = next_state
            
            next_state = normalize_state(next_state, ob_space, s_size)

            # Train
            target = reward
            if not done:
                target = (reward)# + 0.95 * np.amax(model.predict(next_state)[0]))
            
            print("Scaled reward: " + str(target))
            
            target_f = model.predict(state)
            print("agent learning" + str(target_f))
            target_f[0][action] = target
            print("agent new learning" + str(target_f))
            history = model.fit(state, target_f, epochs=1, verbose=0)
            
            # observer: Observe states in neural network
            # if there is no change of the loss function within 10 steps and epsilon_max < 0.5
            # then stop the execution. It detects changes within 0.01
            if ACTIVATE_OBSERVER:
                observerData.append(history.history['loss'][0])
                observerCounter += 1
                if(observerCounter > 10):
                    observerCounter = 10
                    observerData.pop(0)
                    avg = reduce(lambda x, y: x + y, observerData) / len(observerData)
                    indata = list(map(lambda x: abs(x-avg) < 0.01, observerData))
                    complete = reduce(lambda x, y: x and y, indata)
                    if complete and epsilon_max < 0.5:
                        print("Accuracy: " + str(history.history['acc'][0]))
                        print("Network is trained - exit")
                        os._exit(0)

            #rewardsum += reward
            if epsilon > EPSILON_MIN: epsilon *= epsilon_decay
            
            #rewards.append(reward)
            
            if args.output:
                with open(args.output, 'a') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow([reward, action, episode,currentScenario])
                csvFile.close()
            
            #for ap in range(0, aps):
            #    ifaceaction = int(action / (pow(numChannels, ap)))
            #    ifaceaction = ifaceaction % numChannels
            #    actions[ap].append(ifaceaction)
            
            print ("Reward: " + str(reward))
            print ("GameOver: " + str(done))
            #print ("State: " + str(state))
            #print ("Channel selection:" + str(action))
            
            state = next_state
            
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
            
            currentScenario += 1
            if currentScenario >= SCENARIOS:
                currentScenario = 0
            
            run += 1
            
            # next episode if enough steps, if enough episodes -> exit
            # store model and internal states on change of episode
            if steps <= run:
                if args.trainingfile:
                    model.save_weights(args.trainingfile)
                    with open(args.trainingfile + '.var', 'w') as f:  # Python 3: open(..., 'wb')
                        f.write(json.dumps({
                            'lastreward' : lastreward.tolist(),
                            'minreward' : minreward.tolist(),
                            'maxreward' : maxreward.tolist(),
                            'lastaction' : lastaction.tolist()
                        }))
                if args.episodes and int(args.episodes) <= episode:
                    os._exit(1)
                else:
                    break
            
        episode += 1
