
from sources.source_unity import source_unity
import numpy as np

##### SOURCE UNITY CONTINUOUS CATCHER
class source_unity_continuouscatcher( source_unity ):

    ### __INIT__
    def __init__( self ):

        source_unity.__init__( self , "continuouscatcher" )

    ### INFORMATION
    def num_actions( self ): return 1
    def range_actions( self ): return 1

    ### MAP KEYS
    def map_keys( self , actn ):

        actn = np.clip( actn, -1, 1)
        return np.expand_dims(actn,0)

    ### PROCESS OBSERVATION
    def process( self , obsv ):

        return obsv
