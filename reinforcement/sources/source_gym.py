
from sources.source import source
import importlib
import gym
from gym import wrappers

##### SOURCE PYGAME
class source_gym( source ):

    ### __INIT__
    def __init__( self , game ):

        source.__init__( self )
        self.env = gym.make( game )
        #self.env = wrappers.Monitor(self.env, ".") #record

    ### START SIMULATION
    def start( self ):

        obsv = self.env.reset()
        return self.process( obsv )

    ### MOVE ONE STEP
    def move( self , actn ):

        obsv , rewd , done, info = self.env.step( self.map_keys( actn ) )
        self.env.render()
        return self.process( obsv ) , rewd , done
