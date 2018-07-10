# Credits to the code in 'sources/carla/Environment': https://github.com/GokulNC/Setting-Up-CARLA-RL
from sources.carla.Environment.carla_environment_wrapper import CarlaEnvironmentWrapper as CarlaEnv
from sources.source import source
import signal
import sys
import cv2


##### SOURCE CARLA
class source_carla( source ):

    ### __INIT__
    def __init__( self ):

        source.__init__( self )

        self.action = [0.0, 0.0, 0.0] # throttle, steering, brake
        self.continuous = True

        self.env = CarlaEnv( is_render_enabled = False,
                             num_speedup_steps = 10,
                             run_offscreen = False,
                             cameras = ['SceneFinal', 'Depth', 'SemanticSegmentation'],
                             save_screens = False,
                             continuous = self.continuous )

        def signal_handler(signal, frame):
            print('\nProgram closed!')
            sys.exit(0)
        signal.signal(signal.SIGINT, signal_handler)

    ### INFORMATION
    def num_actions( self ): return 3
    def range_actions( self ): return 1

    ### START SIMULATION
    def start( self ):

        self.env.reset()
        observation, reward, done, _ = self.env.step(self.action)

        return self.process( observation['rgb_image'] )

    ### MOVE ONE STEP
    def move( self , actn ):

        observation, reward, done, _ = self.env.step( actn )

        return self.process( observation['rgb_image'] ), reward, done

    ### PROCESS OBSERVATION
    def process( self , obsv ):

        obsv = cv2.resize( obsv , ( 84 , 84 ) )

        return obsv
