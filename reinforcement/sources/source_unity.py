from sources.source import source
import importlib
import numpy as np
from sources.unity.unityagents import UnityEnvironment

##### SOURCE UNITY
class source_unity( source ):

    ### __INIT__
    def __init__( self, game  ):

        source.__init__( self )
        self.env = UnityEnvironment( file_name = "./sources/unity/" + game, worker_id = 0 )

    ### START SIMULATION
    def start( self ):

        brain_name = self.env.brain_names[0]
        brain_info = self.env.reset(True, None)[brain_name]

        obsv = brain_info.states[0]

        return self.process( obsv )

    ### MOVE ONE STEP
    def move( self , actn ):

        brain_name = self.env.brain_names[0]
        brain_info = self.env.step( actn , memory = None, value = None )[brain_name]

        obsv = brain_info.states[0]
        rewd = brain_info.rewards[0]
        done = brain_info.local_done[0]

        return self.process( obsv ) , rewd , done
