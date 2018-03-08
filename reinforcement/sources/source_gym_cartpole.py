
from sources.source_gym import source_gym
import cv2
import numpy as np


##### SOURCE GYM CARTPOLE
class source_gym_cartpole( source_gym ):

    ### __INIT__
    def __init__( self ):

        source_gym.__init__( self , 'CartPole-v0' )

    ### INFORMATION
    def num_actions( self ): return self.env.action_space.shape[0]

    ### MAP KEYS
    def map_keys( self , actn ):

        if actn[0] : return 0
        if actn[1] : return 1

    ### PROCESS OBSERVATION
    def process( self , obsv ):

        return obsv
