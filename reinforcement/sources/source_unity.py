import signal
import sys
import numpy as np
from sources.source import source
from sources.unity.unityagents import UnityEnvironment

##### SOURCE UNITY
class source_unity( source ):

    ### __INIT__
    def __init__( self, game  ):

        source.__init__( self )

        self.env = UnityEnvironment( file_name = "./sources/unity/" + game, worker_id = 0 )
        self.brain_name = self.env.brain_names[0]
        self.brain_initial_info = self.env.reset(True, None)[self.brain_name]

        def signal_handler(signal, frame):
            self.env.close()
            print('\nSocket closed!')
            print('Program closed!')
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

    ### INFORMATION
    def num_actions( self ):

        return self.env.brains[self.brain_name].action_space_size * len(self.brain_initial_info.agents)

    def num_agents( self ):

        return len(self.brain_initial_info.agents)

    ### START SIMULATION
    def start( self ):

        obsv = self.env.reset(True, None)[self.brain_name].states[0]

        return self.process( obsv )

    ### MOVE ONE STEP
    def move( self , actn ):

        actn = np.reshape( actn, [ self.num_agents(), self.num_actions() // self.num_agents() ] )
        brain_info = self.env.step( self.map_keys(actn) , memory = None, value = None )[self.brain_name]

        obsv = brain_info.states[0]
        rewd = brain_info.rewards[0]
        done = brain_info.local_done[0]

        return self.process( obsv ) , rewd , done
