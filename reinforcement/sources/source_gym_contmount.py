
from sources.source_gym import source_gym
import cv2
import numpy as np

##### SOURCE GYM CONTINUOUS MOUNTAIN CAR V0
class source_gym_contmount( source_gym ):

    ### __INIT__
    def __init__( self ):

        source_gym.__init__( self , 'MountainCarContinuous-v0' )

    ### INFORMATION
    def num_actions( self ): return 1
    def range_actions( self ): return abs(self.env.action_space.high[0])

    ### MAP KEYS
    def map_keys( self , actn ):

        return np.clip( actn, self.env.action_space.low[0], self.env.action_space.high[0])

    ### PROCESS OBSERVATION
    def process( self , obsv ):

        obsv = np.squeeze(obsv)

        return obsv
