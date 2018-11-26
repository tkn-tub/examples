#!/usr/bin/env python3

from UniFlexGym.interfaces.uniflex_controller import UniFlexController
#import os,sys,inspect
#current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#parent_dir = os.path.dirname(current_dir)
#sys.path.insert(0, parent_dir) 
from channel_controller import UniflexChannelController

import gym

class Controller(UniFlexController):
    def __init__(self, **kwargs):
        super()
        return
    
    def reset(self):
        return
    
    def execute_action(self, action):
        return
    
    def render():
        return
    
    def get_observationSpace(self):
        return
    
    def get_actionSpace(self):
        return
    
    def get_observation(self):
        return
    
    def get_gameOver(self):
        return
    
    def get_reward(self):
        return
